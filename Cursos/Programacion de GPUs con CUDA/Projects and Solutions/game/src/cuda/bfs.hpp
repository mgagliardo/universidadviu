#ifndef BFS_HPP
#define BFS_HPP

#include <list>
#include <vector>

#include "checkersboard.cpp"

// Kernel unimplemented.
#define TREELEVEL_CPU_LIMIT 1000000
#define TREELEVEL_CPU_LIMIT_TEST 10

struct treenode
{
    CheckersBoard board;
    unsigned int parentIndex;
    int score;
    bool hasChildren;
};

class BFS
{
public:
    BFS();
    ~BFS();

    // Returns the best legal board
    // BFS object may take up a lot of memory between calls to search and releaseMemory
    CheckersBoard search(const CheckersBoard start, const size_t depth, const bool player1, int plyNum);
    
    void buildLevels(const CheckersBoard start, const size_t depth, const bool player1, int plyNum);
    void getLastLevel(treenode** lastLevel, int& lastLevelSize, bool& player1Up);
    CheckersBoard propogateScoresPickWinner(const bool player1);
    // Called by destructor.
    void releaseMemory();

    bool isTerminal(const CheckersBoard& c, bool player1, int plies) const;
    // 2 = p2, 1 = p1, 0 = draw
    char winner(const CheckersBoard& c, int plies) const;

private:
    // Generate a level based on the previous level. Returns the actual number
    // of nodes in the generated nextLevel. player1 indicates that player1 will
    // be making the move for the next ply
    // The array passed into nextStates must be large enough to hold all
    // possible
    // states generated. This function will not check.
    // Inputs:
    //   baseStates[]
    //   numBaseStates
    //   nextStates[]
    //   player1
    // Outputs:
    //   numNextStates
    size_t generate(treenode * const baseStates, size_t numBaseStates, treenode *nextStates, size_t numNextStates, bool player1);


    // Function for doing the actual generation work on the CPU
    size_t generateCPU(treenode * const baseStates, size_t numBaseStates, treenode *nextStates, bool player1);

    // Generate a set of indices of legal moves for a particular board and piece.
    std::list<unsigned int> possibleSimpleMoves(const CheckersBoard& inBoard, unsigned int pieceIndex, bool player1) const;
    std::list<unsigned int> possibleTakeoverMoves(const CheckersBoard& inBoard, unsigned int pieceIndex, bool player1) const;

    // Generate the nodes resulting from available simple moves and takeover moves.
    std::vector<CheckersBoard> simpleMoveBoards(const CheckersBoard& inBoard, unsigned int pieceIndex, bool player1) const;
    std::vector<CheckersBoard> takeoverChainBoards(const CheckersBoard& inBoard, unsigned int pieceIndex, bool player1) const;


    // Inputs:
    //   baseStates[]
    //   numBaseStates
    //   player1
    // Outputs:
    //   numNextStates
    size_t calcNextLevelSize(treenode *baseStates, size_t numBaseStates, bool player1);

    // Apply the score function to the bottom level of the tree, which has no children.
    // The function can look up what player it should be considering
    void evaluateLeaves();
    
    // Apply a scoring heuristic to a board.
    int utility(const CheckersBoard& c, int plyNum) const;

    // return the selected board
    void propogateScores(treenode *level, treenode *childLevel, size_t levelSize, size_t childLevelSize, bool player1, int levelPlyNum);
    
    bool canMove(const CheckersBoard& c, bool player1) const;

    // Return the current time in milliseconds.
    float getMillis() const;

    // The ply of the root of this tree
    int startingPly_;
    vector<treenode*> treeLevels_;
    vector<size_t> treeLevelSizes_;
    vector<bool> player1Moving_;

};

#endif
