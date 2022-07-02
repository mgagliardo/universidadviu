#include <iostream>
#include "kernel.hu"
#include "math.h"
#define NUMOFFIRSTLAYERBOARDS 20
#define MAXDEPTH 5
#define SHAREDBOARDS (20*MAXDEPTH)

#if defined(__CUDA_ARCH__)
   #define CUDAORCPP __device__
#else
   #define CUDAORCPP inline
#endif

#if 0
    #define OUTPUT(x) OUTPUT x
#else
    #define OUTPUT(x)
#endif

__device__ __forceinline__ int scoreABoard(const CheckersBoard &board)
{
    int p1PieceVal = board.p1WeightedMen() + board.p1NumKings() * 15;
    int p2PieceVal = board.p2WeightedMen() + board.p2NumKings() * 15;

    float scoreMultiplier = INT_MAX / int(180);
    float sign = p1PieceVal - p2PieceVal;
    if(sign == 0) return 0;
    sign /= abs(sign);

    return  -int((float(p1PieceVal) / float(p2PieceVal)) * scoreMultiplier);
    
}
__device__ __forceinline__ void potentialHopIndex(const int moveNumber, const unsigned int currentIndex, unsigned int& hop1, unsigned int& hop2)
{
    hop1 = hop2 = 100;//max int
    int rowDirection = moveNumber>0?1:-1;
    int colDirection = abs(moveNumber)==1?-1:1;

    unsigned int row = currentIndex/8;
    unsigned int col = currentIndex%8;

    row += rowDirection;
    col += colDirection;
    if(row<=8 && col<=8) //compute first hop
        hop1 = row*8 + col;

    row += rowDirection;
    col += colDirection;
    if(row<=8 && col<=8) //compute second hop
        hop2 = row*8 + col;
}

__device__ void captureChain(CheckersBoard* deviceBoards, const int maxBoardCount, int* memoryStackPointer, const moveArgs& attackArgs)
{
    #define SIZEATTACKSTACK 20
    moveArgs attackStack[SIZEATTACKSTACK];
    int attackStackDepth = 1;    //put first thing on stack
    attackStack[0] = attackArgs;
    
    while(attackStackDepth>0)
    {
        CheckersBoard board = *(attackStack[attackStackDepth-1].board);
        int squareIndex = attackStack[attackStackDepth-1].startIndex;
        int captureThis = attackStack[attackStackDepth-1].hop1;
        int moveHere = attackStack[attackStackDepth-1].hop2;
        attackStackDepth--;
        
        bool kinged = board.p1King(squareIndex);
        board.captureMove(moveHere, captureThis, squareIndex); //some one wants us to attack
        
        squareIndex = moveHere; //we have progressed to this location
        bool foundACapture = false;
        kinged = (!kinged && board.p1King(squareIndex));//if we just got kinged we need to stop

        if(!kinged) //if It wasn't kinged on the last move
        {
            for(int moveNumber = -2; moveNumber <= 2; moveNumber++) //search moves
            {
                if (moveNumber == 0)
                    moveNumber++;
                unsigned int hop1, hop2;
                potentialHopIndex(moveNumber, squareIndex, hop1, hop2); //can I do this move?
                if(board.canMove(hop2, squareIndex, true)) //if yes, do it
                {
                    foundACapture = true;
                    attackStack[attackStackDepth].board = &board;
                    attackStack[attackStackDepth].startIndex = squareIndex;
                    attackStack[attackStackDepth].hop1 = hop1;
                    attackStack[attackStackDepth].hop2 = hop2;
                    attackStackDepth++;
                }
            }
        }
        if(kinged || !foundACapture) //if done making moves for this path
        {
            int indexToWriteTo = atomicAdd(memoryStackPointer, 1);
            board.swapPlayers();
            deviceBoards[indexToWriteTo] = board;
        }  
    }  
}

__device__ void genNext(CheckersBoard* deviceBoards, const int maxBoardCount, const CheckersBoard* boardToExpand, int* memoryStackPointer, const unsigned int squareIndex){
    CheckersBoard boardCpy = *boardToExpand;
    if(boardCpy.p1NumPieces() == 0)
        return;
    __shared__ int foundAttack[1];
    __shared__ moveArgs movesFound[32];
    __shared__ int movesStack[1];

    if(squareIndex == 1)
    {
        foundAttack[0] = 0;//initilize
        movesStack[0] = 0;
    }
      
    __syncthreads(); //make foundAttack update

    moveArgs possibleMoves[4];
    int possibleMovesIndex = 0;
    int moveType = 0;
    if(boardCpy.p1Piece(squareIndex))
    {    
        /*
            I believe that I am spending most of my kernel's time in this loop
        */
        for(int moveNumber = boardCpy.p1King(squareIndex)?-2:1; moveNumber <= 2; moveNumber++)
        {
            if (moveNumber == 0)
                moveNumber++;
            unsigned int hop1,hop2;
            potentialHopIndex(moveNumber, squareIndex, hop1, hop2);
            bool canAttack = boardCpy.canMove(hop2, squareIndex, true);
            bool addIt = (moveType == 0 && boardCpy.canMove(hop1, squareIndex, true)) || canAttack;
            if(addIt)
            {
                if(canAttack && moveType == 0)
                {
                    foundAttack[0] = 1; //tell the world about the attack           
                    moveType = 1;//set to attack
                    possibleMovesIndex = 0; //reset the potential moves because we will be attacking
                }
                possibleMoves[possibleMovesIndex].board = boardToExpand;
                possibleMoves[possibleMovesIndex].startIndex = squareIndex;
                possibleMoves[possibleMovesIndex].hop1 = hop1;
                possibleMoves[possibleMovesIndex].hop2 = hop2;
                possibleMovesIndex++;
            }     
        }
    }
    __syncthreads(); //make foundAttack update

    if(possibleMovesIndex>0 && foundAttack[0] == moveType) //copy moves found into shared
    {
        int indexToWriteTo = atomicAdd(movesStack, possibleMovesIndex);
        for(int i = 0; i < possibleMovesIndex; i++)
            movesFound[indexToWriteTo+i] = possibleMoves[i];
    }
    
    __syncthreads();
    if(threadIdx.x<movesStack[0]) //assign a thread to each move
    {
        if(foundAttack[0] == 0) //no one found an attack
        {
            int indexToWriteTo = atomicAdd(memoryStackPointer, 1);
            moveArgs argCpy = movesFound[threadIdx.x];            
            CheckersBoard boardToWorkWith = *(argCpy.board);
            boardToWorkWith.move(argCpy.hop1, argCpy.startIndex);
            boardToWorkWith.swapPlayers();            
            deviceBoards[indexToWriteTo] = boardToWorkWith;
        }
        else
        {  
            captureChain(deviceBoards, maxBoardCount, memoryStackPointer, movesFound[threadIdx.x]);         
        }    
    }
}
__device__ void MAXreduction(int *putMaxInIndexZero, const unsigned int size)   //only good for size less then block size*2
{
    int threadId = threadIdx.x;
    /*
        We might be able to speed this up by setting stride =  size. But I can't decide if that will work.
    */

    for(int stride = blockDim.x; stride>=1; stride/=2)                  //divide stride by 2 each time
    {
        __syncthreads();                                                //make sure that we have loaded mem, and finished computing a partial sum
        if(threadId < stride && threadId + stride < size)               //avoid adding things that are out of range
            putMaxInIndexZero[threadId] = (int)fmaxf(putMaxInIndexZero[threadId], putMaxInIndexZero[threadId + stride]);
    }
}

__global__ void getNextBoard(const CheckersBoard* boardsToExpand, CheckersBoard* nextBoard, int* thisBoardsScore)
{
    __shared__ CheckersBoard sharedBoards[SHAREDBOARDS];//boards being generated
    __shared__ int topScores[SHAREDBOARDS];           //scores of boards in NEGAMAX
    __shared__ int subStackLastIndex[MAXDEPTH];         //index by depth of search, holds last index of a level of boards
    __shared__ int pointInSubStack[MAXDEPTH];           //index by depth of search, holds the index currently being expanded on at this depth
    __shared__ int nextFreeIndex[1];                    //next index in sharedBoards that can be written too. Should be atomicly incremented if adding data
    __shared__ int depth[1];                            //the current depth, number of moves made
    __shared__ volatile bool done[1];                   //flag to exit loops
    __shared__ volatile bool exitCondition[1];          //flag to end program
    
    for(int i = threadIdx.x;i < SHAREDBOARDS;i+=blockDim.x)
    {
        sharedBoards[i].clear();
        topScores[i] = -INT_MAX;
    }
    if(threadIdx.x < MAXDEPTH)
    {  
        subStackLastIndex[threadIdx.x] = 0; 
        pointInSubStack[threadIdx.x] = 0;
    }
    if(threadIdx.x == 0)
    { 
        sharedBoards[0] = boardsToExpand[blockIdx.x];   //load the first board
        nextFreeIndex[0] = 1;                           //denote next free memory location in the sharedBoards and topScores arrays
        depth[0]=1;                                     //current depth is 1 because we are going to generate lvl 1 next
        done[0]=false;
        exitCondition[0]=false;
    }
    __syncthreads();                                    //get memory all set up
#define OUTERLOOP 1
#define SCORING 1
#define SORTING 1
#if OUTERLOOP
    do
    {
#endif
  //      int64_t start_time = clock64();
        do                                                                                                                                       //gen next layers
        {   
            unsigned int squareIndex = ((threadIdx.x%32) + 1)*2-1 - (threadIdx.x/4)%2; //index into the right location
            genNext(sharedBoards, SHAREDBOARDS, sharedBoards+subStackLastIndex[depth[0]]-pointInSubStack[depth[0]], nextFreeIndex, squareIndex);//produce all boards possible from the current board
            __syncthreads();
            if(threadIdx.x == 0)
            {    
                //OUTPUT("Gen lvl=%i size =%i\n",depth[0],(nextFreeIndex[0]) - (subStackLastIndex[depth[0]]+1));
                if(depth[0]<MAXDEPTH-1 && nextFreeIndex[0] < SHAREDBOARDS-20)                                                                   //we have not made it to full depth yet, continue deeper!
                {
                    depth[0]++;
                    subStackLastIndex[depth[0]] = nextFreeIndex[0] - 1;
                    done[0] = false;                                                   //progress to new level
                }
                else
                {
                    if(nextFreeIndex[0] >= SHAREDBOARDS-20)//if this is getting hit we need bigger array to make max depth
                    {
                       OUTPUT("Shared boards limit hit\n");
                    }
                    done[0] = true;                                                                                                              //at full depth
                }
            }
            __syncthreads();
        }while(!done[0]);
     //   int64_t stopGenTime = clock64();

#if SCORING
        for(int i =  threadIdx.x + subStackLastIndex[depth[0]]+1;i < nextFreeIndex[0];i+=blockDim.x)                                             //score all boards at the bottom level
        {
            topScores[i] = scoreABoard(sharedBoards[i]);
        }

        __syncthreads();
      //  int64_t stopScoretime = clock64();
        
#endif
#if SORTING
        do{       
            MAXreduction(topScores+subStackLastIndex[depth[0]]+1, (nextFreeIndex[0]) - (subStackLastIndex[depth[0]]+1));//find the max score for the level
            __syncthreads();
            if(threadIdx.x == 0)
            {
                //OUTPUT("Write to %i\n",subStackLastIndex[depth[0]]-pointInSubStack[depth[0]]);
                if((nextFreeIndex[0]) - (subStackLastIndex[depth[0]]+1)>0)
                {
                    topScores[subStackLastIndex[depth[0]]-pointInSubStack[depth[0]]]=-topScores[subStackLastIndex[depth[0]]+1];//copy up a level
                }
                else //the opponent was not able to make a move, we win
                {
                   // OUTPUT("%i No More Moves for player%i in %i turns\n",blockIdx.x, sharedBoards[subStackLastIndex[depth[0]]-pointInSubStack[depth[0]]].getP1Up()?1:2,depth[0]);
                    topScores[subStackLastIndex[depth[0]]-pointInSubStack[depth[0]]] = scoreABoard(sharedBoards[subStackLastIndex[depth[0]]-pointInSubStack[depth[0]]]);
                }
                nextFreeIndex[0] = subStackLastIndex[depth[0]]+1;                                       //reset stack, essentially we don't need the boards we just looked at so delete them
                pointInSubStack[depth[0]]++;                                                            //bump to next parent node

                if(subStackLastIndex[depth[0]]-pointInSubStack[depth[0]]<=subStackLastIndex[depth[0]-1])//we have exhausted all parents at the current level
                {
                    pointInSubStack[depth[0]] = 0; 
                    subStackLastIndex[depth[0]] = 0;                                                    //reset just for good luck, probably don't need to

                    depth[0]--;                                                                         //move up a level
                    if(depth[0] != 1)      
                        done[0] = false;                                                                //since the level we are now at has stuff to be done, like finding the best possible score, we will loop and do it
                    else                                                                                //return condition we just finished scoring the nextMoves
                    {
                        done[0] = true;
                        exitCondition[0]=true;                                                          //we have found the best move!~!!!
                        int bestScore = -INT_MAX;
                        CheckersBoard bestBoard;
                        //OUTPUT("Inspecting index %i to %i\n",subStackLastIndex[depth[0]]+1, nextFreeIndex[0]);
                        for(int i = subStackLastIndex[depth[0]]+1; i < nextFreeIndex[0]; i++)           //slow loop probably can be sped up, but we need to know what board was the best...its not just a numbering as in the other lvls
                        {
                            //OUTPUT("score = %f\n",topScores[i]);
                            if(topScores[i] > bestScore)
                            {
                                //OUTPUT("Board set\n");
                                bestScore = topScores[i];
                                bestBoard = sharedBoards[i];
                            }
                        }
                        nextBoard[blockIdx.x] = bestBoard; //the best next board after the head of block
                        thisBoardsScore[blockIdx.x] = -bestScore; //score of the head board of this block...not the score of the bestBoard
                    }
                }
                else
                {
                    done[0] = true;
                }
            }
            __syncthreads();
        }while(!done[0]);
      //  int64_t stopSortTime = clock64();
      /*  if(threadIdx.x==0)
        {
            OUTPUT("GenNext time = %lli\n",stopGenTime-start_time);
            OUTPUT("Scoring time = %lli\n",stopScoretime-stopGenTime);
            OUTPUT("Sorting time = %lli\n",stopSortTime-stopScoretime);
        }*/
#endif
#if OUTERLOOP
    }while(!exitCondition[0]);
#endif
}


CheckersBoard getOptimalNextBoard(CheckersBoard startBoard)//contains the origin board in index = 0
{
    clock_t startTime = clock();
    int firstLayerStack = 0;
    CheckersBoard* firstLayerBoards;
    CheckersBoard* secondLayerBoards;
    int* firstLayerScores;
    bool wasP1Up = startBoard.getP1Up();
    if(!startBoard.getP1Up())
        startBoard.swapPlayers(); //BFS code expects p1 to be in p1 seat, DFS with rotate each time...
    BFS bfs;
    bfs.buildLevels(startBoard, 6, wasP1Up, 0); //given a startBoard, produce leaf boards at a depth so I can have lots of blocks
    treenode* lastLevel = NULL;
    bool player1Up = false;
    bfs.getLastLevel(&lastLevel,firstLayerStack,player1Up); //extract the leaf boards and establish a block count
    player1Up = !player1Up;
    cout<<"Block Count = "<<firstLayerStack<<" p1 was up = "<<wasP1Up<<" after layers p1 is "<<player1Up<<endl;
    firstLayerBoards = (CheckersBoard*)malloc(firstLayerStack*sizeof(CheckersBoard));
    secondLayerBoards = (CheckersBoard*)malloc(firstLayerStack*sizeof(CheckersBoard));
    firstLayerScores = (int*)malloc(firstLayerStack*sizeof(int));
    for(int i = 0; i<firstLayerStack;i++)
    {
        CheckersBoard boardCpy = lastLevel[i].board;
        if(!player1Up)
            boardCpy.swapPlayers(); //translate due to the fact that the BFS code does not rotate the board, while the DFS code does
        boardCpy.setP1Up(player1Up);
        firstLayerBoards[i] = boardCpy;
    }
    //firstLayerBoards[0] = startBoard;
    //firstLayerStack = 1;

    

    CheckersBoard* boardsToExpand;
    CheckersBoard* nextBoard;
    int* thisBoardsScore;
    dim3 dim_grid, dim_block;
    dim_block.x = 32;
    dim_grid.x = firstLayerStack;
    dim_grid.y = dim_grid.z = dim_block.y = dim_block.z = 1;
    cudaError_t cuda_ret = cudaMalloc((void**)&boardsToExpand, firstLayerStack*sizeof(CheckersBoard));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&nextBoard, firstLayerStack*sizeof(CheckersBoard));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&thisBoardsScore, firstLayerStack*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Fail Sync");
    
    cuda_ret = cudaMemcpy(boardsToExpand, firstLayerBoards, firstLayerStack*sizeof(CheckersBoard), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Fail Sync");
    OUTPUT("Launch now!\n");
    
    getNextBoard<<<dim_grid, dim_block>>>(boardsToExpand, nextBoard, thisBoardsScore); //launch the DFS kernel

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    cuda_ret = cudaMemcpy(secondLayerBoards, nextBoard, firstLayerStack*sizeof(CheckersBoard), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(firstLayerScores, thisBoardsScore, firstLayerStack*sizeof(int), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    
    cudaFree(boardsToExpand);
    cudaFree(nextBoard);
    cudaFree(thisBoardsScore);
    
    for(int i = 0; i<firstLayerStack;i++)
    {
        if(player1Up)
            lastLevel[i].score = -firstLayerScores[i]; //translate scores from the DFS to the BFS
        else
            lastLevel[i].score = firstLayerScores[i];
    }
    free(firstLayerBoards);
    free(secondLayerBoards);
    free(firstLayerScores);

    CheckersBoard bestBoard = bfs.propogateScoresPickWinner(wasP1Up); //propogate the scores using the BFS
    if(wasP1Up)
        bestBoard.swapPlayers(); //rotate the board 
    bestBoard.setP1Up(!wasP1Up);
    clock_t finishTime = clock();
    cout<<"Runtime = "<<((double)finishTime - startTime)/CLOCKS_PER_SEC<<endl;
    return bestBoard;

    
}















