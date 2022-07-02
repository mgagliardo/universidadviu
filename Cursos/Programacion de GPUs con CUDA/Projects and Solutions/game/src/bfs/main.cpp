#include "../sequential/checkersboard.cpp"
#include "bfs.hpp"

using namespace std;

int main(void)
{


  CheckersBoard c;
  BFS gameChecker;
  bool player1 = true;
  int plyNum = 0;
  while (!gameChecker.isTerminal(c, player1, plyNum)) {
    BFS bfs;
    c = bfs.search(c, 6, player1, plyNum);
    c.dump(cout);
    cout << endl;
    player1 = !player1;
    ++plyNum;
    //char b;
    //gets(&b); 
  }

char w = gameChecker.winner(c, plyNum);

if (w == 0) cout << "draw" << endl;
if (w == 1) cout << "player1 wins!" << endl;
if (w == 2) cout << "player2 wins!" << endl;



/*
  //CheckersBoard c;
  CheckersBoard c(0x00040E18, 0x66881000, 0);
  cout << "starting board before bfs search call:" << endl;
  c.dump(cout);
  cout << endl;
  BFS bfs;
  cout << "BFS search:" << endl;
  CheckersBoard result = bfs.search(c, 2, false, 1);
  cout << endl << "result after BFS search:" << endl;
  result.dump(cout);
  //result = bfs.search(result, 2, false, 1);
  //cout << endl;
  //result.dump(cout);
*/

}
