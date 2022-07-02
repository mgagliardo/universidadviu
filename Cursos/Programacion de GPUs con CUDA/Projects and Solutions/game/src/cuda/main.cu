/******************************************************************************
*cr
*cr         (C) Copyright 2010-2013 The Board of Trustees of the
*cr                        University of Illinois
*cr                         All Rights Reserved
*cr
******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <cassert>

#include "support.h"
#include "kernel.cu"
using namespace std;
int main(int argc, char* argv[])
{
    Timer timer;
    CheckersBoard nextBoard;
    CheckersBoard startBoard;//(0x00002DFF,0xFF930000,0);
    if(!startBoard.getP1Up())
        startBoard.swapPlayers();
    startBoard.dump(cout);
    if(startBoard.getP1Up())
        startBoard.swapPlayers();
    cout<<endl<<endl;
    startTime(&timer);
    clock_t startClock = clock();
    for(int i = 0; i < 100 && startBoard.p1NumPieces()>0; i++)
    {
        nextBoard = getOptimalNextBoard(startBoard);        

        startBoard = nextBoard;
        cout<<i<<" Player 1 up = "<<nextBoard.getP1Up();
        if(!nextBoard.getP1Up())
            nextBoard.swapPlayers();
        cout<<" Player 1 pieces "<<nextBoard.p1NumPieces()<<" vs "<<nextBoard.p2NumPieces()<<endl;
        nextBoard.dump(cout);
        cout<<endl<<endl;
    }
    clock_t stopClock = clock();
    stopTime(&timer); printf("Total Run time = %f s  vs %f\n", elapsedTime(timer),(float)(stopClock-startClock)/CLOCKS_PER_SEC);
    return 0;
}

