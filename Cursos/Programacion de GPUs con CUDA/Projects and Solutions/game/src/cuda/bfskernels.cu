
#include "bfskernels.hu"

using namespace std;

__device__ int utility(CheckersBoard board)
{
  // This method will calculate and return the utility of the given board.

  // max value here is if every man is a king = 12*15 = 180, and the other 
  // player has no pieces = 0
  int p1PieceVal = board.p1WeightedMen() + board.p1NumKings() * 15;
  int p2PieceVal = board.p2WeightedMen() + board.p2NumKings() * 15;

  float scoreMultiplier = INT_MAX / int(180);
  float sign = p1PieceVal - p2PieceVal;
  if(sign == 0) return 0;
  // Get the sign.
  sign /= abs(sign);

  return  int((sign * float(p1PieceVal) / float(p2PieceVal)) * scoreMultiplier);
}

__global__ void prepareParents(treenode* level, unsigned int size, bool p1)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int id = bid * blockDim.x + tid;

  // Initialize variables.
  treenode myparent;
  treenode* myparentAddress;
  // One treenode in the upper level per thread.
  if(id < size)
  {
    // Get treenode from upperlevel.
    myparentAddress = &level[id];
    myparent = *myparentAddress;

    // Decide how to initialize score.
    int score = 0;
    if(!myparent.hasChildren) score = utility(myparent.board);
    else if(p1) score = INT_MIN;
    else score = INT_MAX;

    // Set score in parent.
    myparentAddress->score = score;
  }
}


// Every thread is a treenode from the upper level that is looking for
// its child with the maximum utility.
__global__ void scoreCascade(treenode* lowerLevel, treenode* upperLevel,
    unsigned int lowerSize, unsigned int upperSize, bool p1)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int id = bid * blockDim.x + tid;

  // have each thread select a treenode from the lower level.
  treenode mynode;
  if(id < lowerSize)
  {
    mynode = lowerLevel[id];
  }

  __syncthreads();

  if(id < lowerSize)
  {
    treenode* parent = &upperLevel[mynode.parentIndex];
    bool didSwap = false;
    while(!didSwap)
    {
      int parentScore = parent->score;
      // If we are player one and our score is smaller than the current score in
      // the parent, break. We are trying to maximize.
      if(p1 && mynode.score <= parentScore) break;
      // If we are player two and our score is greater than the current score in
      // the parent, break. We are trying to minimize.
      if(!p1 && mynode.score >= parentScore) break;
      int old = atomicCAS(&(parent->score),parentScore, mynode.score);
      if(old == parentScore)
      {
        didSwap = true;
      }
    }
  }
}




__global__ void evaluateLeavesGPU(treenode* leaves, unsigned int numLeaves)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int id = bid * blockDim.x + tid;

  if(id < numLeaves)
  {
    // Evaluate leaves in parallel.
    treenode* mynode = &leaves[id];
    CheckersBoard myBoard = mynode->board;
    int util = utility(myBoard);
    mynode->score = util;
  }
}


  // Some constants used by the BFS generate code
const unsigned PSMR_SIZE = 4;
const unsigned PCCR_SIZE = 8;
const unsigned TEST_SIZE = 12;
const unsigned SMR_SIZE = PSMR_SIZE * 12;

__device__ void simpleMoveBoards(CheckersBoard* psmr, unsigned* psmr_back, CheckersBoard baseBoard, unsigned char pieceIndex,  bool player1){

  unsigned char pieceRow = pieceIndex / 8;
  unsigned char pieceCol = pieceIndex % 8;

  // Loop over all possible target indices 
  for (char rowMod = -1; rowMod < 2; rowMod += 2) {
    for (char colMod = -1; colMod < 2; colMod += 2) {
      unsigned char targetRow = char(pieceRow) + rowMod;
      unsigned char targetCol = char(pieceCol) + colMod;
      if ((targetRow < 8) && (targetCol < 8)) {
        unsigned char targetIndex = targetRow * 8 + targetCol;
        // move is only valid if the target index is empty
        if (!baseBoard.piece(targetIndex)) {
          // If the piece is not a king, it can't capture "behind"
          if ( (player1  && baseBoard.p1Man(pieceIndex)  && (targetIndex > pieceIndex)) ||
               (!player1 && baseBoard.p2Man(pieceIndex)  && (targetIndex < pieceIndex)) ||
               (player1  && baseBoard.p1King(pieceIndex)                              ) ||
               (!player1 && baseBoard.p2King(pieceIndex)                              )   ) {
            CheckersBoard newBoard = baseBoard;
            newBoard.move(targetIndex, pieceIndex);
            psmr[(*psmr_back)++] = newBoard;
          }
        }
      }
    }
  }

}




// Result boards from capture chains on baseBoard at piece baseIndex go into pccr, with the size in pccr_back.
// stack is scratch space unique to this thread
__device__ void captureChainBoards(CheckersBoard* pccr,  unsigned* pccr_back, 
                                   CheckersBoard* boardStack, unsigned char* pieceStack,
                                   CheckersBoard baseBoard, unsigned char baseIndex, bool player1){
  unsigned int stackIndex = 0; // Compiler crashes if this is an unsigned char. Real talk
  // Take the first board and add it to the stack
  unsigned char pieceRow = baseIndex / 8;
  unsigned char pieceCol = baseIndex % 8;
  for (char rowMod = -2; rowMod < 3; rowMod += 4) {
    for (char colMod = -2; colMod < 3; colMod += 4) {
      unsigned char targetRow = char(pieceRow) + rowMod;
      unsigned char targetCol = char(pieceCol) + colMod;
      if ((targetRow < 8) && (targetCol < 8)) {
        unsigned char targetIndex = targetRow * 8 + targetCol;
        // Check that the target index is empty
        if (!baseBoard.piece(targetIndex)) {
          // only kings can move backwards   
          if ( (player1  && baseBoard.p1Man(baseIndex)  && (targetIndex > baseIndex)) ||
               (!player1 && baseBoard.p2Man(baseIndex)  && (targetIndex < baseIndex)) ||
               (player1  && baseBoard.p1King(baseIndex)                              ) ||
               (!player1 && baseBoard.p2King(baseIndex)                              )   ) {
            unsigned char captureIndex = (targetIndex + baseIndex) / 2;
            // check that the capture index is occupied by an opposing piece
            if ( (player1  && baseBoard.p2Piece(captureIndex)) ||
                 (!player1 && baseBoard.p1Piece(captureIndex))    ) {
              // This is a valid capture move
              CheckersBoard newBoard(baseBoard);
              newBoard.captureMove(targetIndex, captureIndex, baseIndex);
              boardStack[stackIndex] = newBoard;
              pieceStack[stackIndex] = targetIndex;
              ++stackIndex;
            }
          }
        }
      }
    }
  } 

  // While there are boards in the stack, take each board and generate the children
  // that result from a takeover
  while (stackIndex != 0) {
    --stackIndex;
    CheckersBoard testBoard = boardStack[stackIndex];
    unsigned char testPiece = pieceStack[stackIndex];
    // If there aren't any capture moves resulting from this board, it ends a captur
    // chain
    
    pieceRow = testPiece / 8;
    pieceCol = testPiece % 8;
    bool testBoardEndsCaptureChain = true;
    
    for (char rowMod = -2; rowMod < 3; rowMod += 4) {
      for (char colMod = -2; colMod < 3; colMod += 4) {
        unsigned char targetRow = char(pieceRow) + rowMod;
        unsigned char targetCol = char(pieceCol) + colMod;
        
        if ((targetRow < 8) && (targetCol < 8)) {
          unsigned char targetIndex = targetRow * 8 + targetCol;
          // Check that the target index is empty
          
          if (!testBoard.piece(targetIndex)) {
            // only kings can move backwards   
            
            if ( (player1  && testBoard.p1Man(testPiece)  && (targetIndex > testPiece)) ||
                 (!player1 && testBoard.p2Man(testPiece)  && (targetIndex < testPiece)) ||
                 (player1  && testBoard.p1King(testPiece)                             ) ||
                 (!player1 && testBoard.p2King(testPiece)                             )   ) {
              unsigned char captureIndex = (targetIndex + testPiece) / 2;
              // check that the capture index is occupied by an opposing piece
              
              if (( player1 && testBoard.p2Piece(captureIndex)) || 
                  (!player1 && testBoard.p1Piece(captureIndex))    ) {
                // This is a valid capture move
                testBoardEndsCaptureChain = false;
                CheckersBoard newBoard(testBoard);
                newBoard.captureMove(targetIndex, captureIndex, testPiece);
                boardStack[stackIndex] = newBoard;
                pieceStack[stackIndex] = targetIndex;
                ++stackIndex;
                
              }
            
            }
          
          }
        
        }
      
      }
    }

    // This piece/board combination may not have generated any more capture moves
    // if it didn't, it ends a capture chain and should be returned as a capture
    // chain result by placing it into pccr
    if (testBoardEndsCaptureChain) {
      pccr[(*pccr_back)++] = testBoard;
    }
  } //while
}

__global__ void generateGPU(treenode *baseStates, unsigned numBaseStates, treenode *nextStates, unsigned *numGeneratedStates, bool player1)
{

  const unsigned tid = threadIdx.x;
  const unsigned bid = blockIdx.x;
  const unsigned  id = bid * blockDim.x + tid;

  // Shared among blockDim.x threads
  // scratch area for accumulating piece & board move lists from a block

  // Piece simple move results maximum vector size
  // piece takeover chain results maximum vector size
  //const unsigned PSMR_SIZE = 4;
  //const unsigned PCCR_SIZE = 8;
  //const unsigned SMR_SIZE = PSMR_SIZE * 12;

  // This will be the scratch area for threads to work in while accumulating
  // moves for each pice in the board.
  // Instead of being written out to global memory for each piece, simple moves
  // are kept in shared memory until the entire board is generated, and only sent out to global if there were no
  // possible takeover moves from this board

  __shared__ CheckersBoard pieceSimpleMoveResults    [GENERATE_BLOCK_SIZE][PSMR_SIZE];
  __shared__ CheckersBoard simpleMoveResults         [GENERATE_BLOCK_SIZE][SMR_SIZE];
  __shared__ CheckersBoard pieceCaptureChainResults  [GENERATE_BLOCK_SIZE][PCCR_SIZE];
  // Scratch space for when the capture chain is being built. Each board in the work list
  // need to go along with whichever piece index is continuing the capture chain
  // FIXME: does this stack have to be bigger?
  __shared__ CheckersBoard testBoardStack            [GENERATE_BLOCK_SIZE][TEST_SIZE];
  __shared__ unsigned char testIndexStack            [GENERATE_BLOCK_SIZE][TEST_SIZE];

  // simple move results. Used for simulating a vector
  unsigned smr_back = 0;
  bool boardHasCaptureChains = false;
  bool boardHasChildren = false;

  // Done by one thread in ther kernel
  if (0 == id) {
    (*numGeneratedStates) = 0; 
  }
  __syncthreads();

  // Clear shared memory
  for (int i = 0; i < PSMR_SIZE; ++i) pieceSimpleMoveResults[tid][i] = CheckersBoard();
  for (int i = 0; i < SMR_SIZE; ++i) simpleMoveResults[tid][i] = CheckersBoard(0x0000FFFF, 0xFFFF0000, 0);
  for (int i = 0; i < PCCR_SIZE; ++i) pieceCaptureChainResults[tid][i] = CheckersBoard(0,0,0); 
  for (int i = 0; i < TEST_SIZE; ++i) testIndexStack[tid][i] = -1;
  for (int i = 0; i < TEST_SIZE; ++i) testBoardStack[tid][i] = CheckersBoard();
  __syncthreads();


  if (id < numBaseStates) {
    // FIXME: are we writing back whether this has children or not?
    const treenode baseNode = baseStates[id];
    const CheckersBoard baseBoard = baseNode.board;

    // loop through all spaces on the board
    for (unsigned char boardIndex = 0; boardIndex < 64; ++boardIndex) {
      bool pieceHasCaptureChains = false;
      unsigned psmr_back = 0;
      unsigned pccr_back = 0;
      if (baseBoard.isOccupiableSpace(boardIndex)) {
        if ( (player1 && (baseBoard.p1Piece(boardIndex) == true)) || (!player1 && (baseBoard.p2Piece(boardIndex) == true)) ) {
          
          // Generate takeover-chain boards for this piece. Pass it a pointer to the scratch space for building its stacks 
          captureChainBoards(pieceCaptureChainResults[tid], &pccr_back, testBoardStack[tid], testIndexStack[tid], baseBoard, boardIndex, player1);
          
          // Update whether this piece generated takeover moves.
          // Likewise, we can update whether the whole board does
          
          pieceHasCaptureChains = (pccr_back != 0);
          boardHasCaptureChains |= pieceHasCaptureChains;
          boardHasChildren |= pieceHasCaptureChains;
          //pieceHasCaptureChains = false;
          //boardHasCaptureChains = false;

          // If the board isn't known to have capture chains, generate simple moves and put them into shared memory
          if (!boardHasCaptureChains) {
            simpleMoveBoards(pieceSimpleMoveResults[tid], &psmr_back, baseBoard, boardIndex, player1);
            boardHasChildren |= (psmr_back > 0);
            for (unsigned i = 0; i < psmr_back; ++i) {
              simpleMoveResults[tid][smr_back++] = pieceSimpleMoveResults[tid][i];
            }
          }
          // If the piece has capture chains, move the resulting boards to global memory
          if (pieceHasCaptureChains) {
            // Request some space in the global array equal to the number of capture moves this piece generated
            unsigned chunkStart = atomicAdd(numGeneratedStates, pccr_back);
            unsigned chunkEnd = chunkStart + pccr_back;
            // Write our values into our chunk of space
            for (unsigned i = chunkStart; i < chunkEnd; ++i) {
              treenode tn = {pieceCaptureChainResults[tid][i - chunkStart], id, 0, 0};
              nextStates[i] = tn;
            }
          }

        }
      }
    }
    // Done going through board pieces
    // If the board hasn't made any capture moves,  move all of the simple moves from shared
    // memory to global memory
    if (!boardHasCaptureChains) {
      unsigned chunkStart = atomicAdd(numGeneratedStates, smr_back);
      unsigned chunkEnd = chunkStart + smr_back;
      for (unsigned i = chunkStart; i < chunkEnd; ++i) {
        treenode tn = {simpleMoveResults[tid][i - chunkStart], id, 0, 0};
        nextStates[i] = tn;
      }
    }
    
    // tell parent boards if they have children or not.
    baseStates[id].hasChildren = boardHasChildren;
  }

}

void scoreCascadeWrapper(treenode* lowerLevel, treenode* upperLevel,
    unsigned int lowerSize, unsigned int upperSize, bool p1)
{
  dim3 dimPrepGrid, dimGrid, dimBlock;
  dimPrepGrid.x = ceil(float(upperSize)/float(BLOCK_SIZE));
  dimGrid.x = ceil(float(lowerSize)/float(BLOCK_SIZE));
  dimBlock.x = BLOCK_SIZE;

  // Void last error.
  cudaGetLastError();

  treenode* d_lowerLevel;
  treenode* d_upperLevel;

  cudaMalloc((void **)&d_lowerLevel, sizeof(treenode)*lowerSize);
  cudaMalloc((void **)&d_upperLevel, sizeof(treenode)*upperSize);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error while allocating memory on the GPU.\n" );

  cudaMemcpy(d_lowerLevel, lowerLevel, sizeof(treenode)*lowerSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_upperLevel, upperLevel, sizeof(treenode)*upperSize, cudaMemcpyHostToDevice);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error while copying to GPU.\n" );

  // Launch kernel to set parent scores correctly.
  prepareParents<<<dimGrid,dimBlock>>>(d_upperLevel, upperSize, p1);
  cudaError_t prepError = cudaGetLastError();
  if ( prepError != cudaSuccess )
    printf("%s\n", cudaGetErrorString(prepError));

  // Launch kernel to propagate child-scores to parents.
  scoreCascade<<<dimGrid,dimBlock>>>(d_lowerLevel, d_upperLevel,
      lowerSize, upperSize, p1);
  cudaError_t error = cudaGetLastError();
  if ( error != cudaSuccess )
    printf("%s\n", cudaGetErrorString(error));

  cudaMemcpy(upperLevel, d_upperLevel, sizeof(treenode)*upperSize, cudaMemcpyDeviceToHost); 
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error while copying to the host.\n" );

  cudaFree(d_lowerLevel);
  cudaFree(d_upperLevel);
}

void evaluateLeavesWrapper(treenode* leaves, unsigned int numleaves)
{
  dim3 dimGrid, dimBlock;
  dimGrid.x = ceil(float(numleaves)/float(BLOCK_SIZE));
  dimBlock.x = BLOCK_SIZE;

  // Void last error.
  cudaGetLastError();

  treenode* d_leaves;
  cudaMalloc((void **)&d_leaves, sizeof(treenode)*numleaves);
  cudaMemcpy(d_leaves, leaves, sizeof(treenode)*numleaves, cudaMemcpyHostToDevice);

  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error while initializing device memory.\n" );

  evaluateLeavesGPU<<<dimGrid,dimBlock>>>(d_leaves, numleaves);

  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error while executing kernel.\n" );

  cudaMemcpy(leaves, d_leaves, sizeof(treenode)*numleaves, cudaMemcpyDeviceToHost);
  cudaFree(d_leaves);
}

unsigned generateWrapper( treenode *baseStates, unsigned numBaseStates,
    treenode *nextStates, unsigned numNextStates, bool player1) {

  //printf("generateWrapper: %d incoming states\n", numBaseStates);
  //for (unsigned i = 0; i < numBaseStates; ++i) {
  //  baseStates[i].board.dump(cout);
  //  printf("\n");
  //}

  //const unsigned MAX_KERNEL = 65536 * GENERATE_BLOCK_SIZE;
 
  // May have to issue multiple kernels.
  //treenode *kernel_nextStates = nextStates;
  //for (unsigned offset = 0; offset < numBaseStates; offset += MAX_KERNEL) {
  //  treenode *kernel_baseStates = baseStates + offset;
  //  unsigned  kernel_numBaseStates = (numBaseStates - offset > MAX_KERNEL) ? MAX_KERNEL : numBaseStates % MAX_KERNEL;
  //  printf("SIM: launch kernel with offset=%d and numBaseStates=%d\n", offset, kernel_numBaseStates);

  //  size_t kernel_numNextStates = 0; // how much was generated by the last kernel
  //  kernel_nextStates += kernel_numNextStates;
  //}


  // Try to allocate GPU memory
  treenode *d_baseStates;
  treenode *d_nextStates;
  // So the GPU can report how many states it ended up creating
  unsigned *d_numGeneratedStates;
  assert(sizeof(numNextStates) == sizeof(*d_numGeneratedStates));



  cudaMalloc( (void**) &d_numGeneratedStates, sizeof(*d_numGeneratedStates));
  cudaError_t error = cudaGetLastError();
  if ( cudaSuccess != error ) {
    printf( "Error while allocating memory for number of generated states on the GPU.\n" );
    printf("%s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaMalloc( (void**) &d_baseStates, numBaseStates * sizeof(treenode));
  if ( cudaSuccess != cudaGetLastError() ) {
    printf( "Error while allocating memory for base states on the GPU.\n" );
    exit(-1);
  }

  cudaMalloc( (void**) &d_nextStates, numNextStates * sizeof(treenode));
  if ( cudaSuccess != cudaGetLastError() ) {
    printf( "Error while allocating memory for generated states on the GPU.\n" );
    exit(-1);
  }

  cudaMemcpy(d_baseStates, baseStates, sizeof(treenode) * numBaseStates, cudaMemcpyHostToDevice);
  if ( cudaSuccess != cudaGetLastError() ) {
    printf( "Error while copying baseStates GPU.\n" );
    exit(-1);
  }

  // Set up the cuda grid
  dim3 dimGrid(1,1,1);
  dimGrid.x = numBaseStates / GENERATE_BLOCK_SIZE;
  //printf("base level size: %d\n", numBaseStates);
  if ((numBaseStates % GENERATE_BLOCK_SIZE) != 0) dimGrid.x++;
  //printf("dimGrid.x: %d\n", dimGrid.x);
  dim3 dimBlock(GENERATE_BLOCK_SIZE, 1, 1);
  //printf("dimBlock.x: %d\n", dimBlock.x);

  cudaDeviceSynchronize();
  if ( cudaSuccess != cudaGetLastError() ) {
    printf( "Error synchronizing before kernel.\n" );
    exit(-1);
  }



  generateGPU<<<dimGrid, dimBlock>>>(d_baseStates, numBaseStates, d_nextStates, d_numGeneratedStates, player1);
  error = cudaGetLastError();
  if ( cudaSuccess != error ) {
    printf( "Error while launching kernel.\n" );
    printf("%s\n", cudaGetErrorString(error));
    exit(-1);
  }


  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if ( cudaSuccess != error ) {
    printf( "Error synchronizing after kernel.\n" );
    printf("%s\n", cudaGetErrorString(error));
    exit(-1);
  }


  // Copy results back to CPU memory
  cudaMemcpy(&numNextStates, d_numGeneratedStates, sizeof(numNextStates), cudaMemcpyDeviceToHost);
  if ( cudaSuccess != cudaGetLastError() ) {
    printf( "Error while copying number of generated states to  CPU.\n" );
    exit(-1);
  }

  cudaMemcpy(baseStates, d_baseStates, sizeof(treenode) * numBaseStates, cudaMemcpyDeviceToHost);
  if ( cudaSuccess != cudaGetLastError() ) {
    printf( "Error while copying base states to CPU.\n" );
    exit(-1);
  }

  cudaMemcpy(nextStates, d_nextStates, sizeof(treenode) * numNextStates, cudaMemcpyDeviceToHost);
  if ( cudaSuccess != cudaGetLastError() ) {
    printf( "Error while copying nextStates to CPU.\n" );
    exit(-1);
  }


  cudaDeviceSynchronize();
  //printf("Number of generated states after kernel invocation: %d\n", numNextStates);

  //for (unsigned i = 0; i < numNextStates; ++i) {
  //  nextStates[i].board.dump(cout);
  //  printf("\n");
  //}

  // Free GPU memory
  cudaFree(d_baseStates);
  cudaFree(d_nextStates);
  cudaFree(d_numGeneratedStates);

  return numNextStates;
}

