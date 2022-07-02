#include <cstdio>
#include <iostream>
#include <cassert>

#include "checkersboard.cpp"
#include "CheckersSearch.hpp"

using namespace std;

int main (int argc, char *argv[])
{
    CheckersBoard c1;
    CheckersSearch cs;
    player p = player1;

    int turns = 100;
    int depth = 9;
    if(argc > 1)
    {
        turns = atoi(argv[1]);
    }
    if(argc > 2)
    {
        depth = atoi(argv[2]);
    }

    cout << "Playing for " << turns << " turns." << endl;

    for(int i = 0; i < turns; i++)
    {
        c1.dump(cout);
        cout << endl;
        c1 = cs.search(&c1, p, depth);
        p = !p;
        if(cs.isTerminal(&c1, p))
        {
            cout << "Game over!" << endl;
            break;
        }
    }
    c1.dump(cout);
    cout << endl;

    cs.pf->printReport();
}
