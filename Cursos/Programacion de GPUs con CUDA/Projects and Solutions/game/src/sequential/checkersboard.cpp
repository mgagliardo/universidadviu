#ifndef CHECKERSBOARD_CPP
#define CHECKERSBOARD_CPP

#include <stdint.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <algorithm>

/*
 *  CHECKERSBOARD.HPP
 *  Interface
 */

#if defined(__CUDA_ARCH__)
   #define CUDAORCPP __device__
#else
   #define CUDAORCPP inline
#endif

class CheckersBoard
{
public:
    CUDAORCPP CheckersBoard();    
    CUDAORCPP CheckersBoard(uint32_t p1, uint32_t p2, uint32_t kings);
    //CUDAORCPP CheckersBoard(const CheckersBoard& toCopy);

    // ==
    CUDAORCPP bool operator==(const CheckersBoard& rhs) const;

    // Print the board to ostream
    inline void dump(std::ostream & o) const; //we don't want this in cuda
    CUDAORCPP void toString(char* boardString);

    // Count the number of pieces the players have on the board
    CUDAORCPP size_t p1NumPieces() const;
    CUDAORCPP size_t p2NumPieces() const;
    CUDAORCPP size_t p1NumMen() const;
    CUDAORCPP size_t p2NumMen() const;
    CUDAORCPP size_t p1NumKings() const;
    CUDAORCPP size_t p2NumKings() const;

    // Returns the sum over all men of (5 + man's row) for a particular player
    CUDAORCPP size_t p1WeightedMen() const;
    CUDAORCPP size_t p2WeightedMen() const;

    // Determine what kind of piece each player has in a particular spot.
    // Input index should be 0-63
    CUDAORCPP bool p1Piece(unsigned int index) const;
    CUDAORCPP bool p1Piece(uint8_t row, uint8_t col) const;
    CUDAORCPP bool p1Man(unsigned int index) const;
    CUDAORCPP bool p1King(unsigned int index) const;
    CUDAORCPP bool p2Piece(unsigned int index) const;
    CUDAORCPP bool p2Piece(uint8_t row, uint8_t col) const;
    CUDAORCPP bool p2Man(unsigned int index) const;
    CUDAORCPP bool p2King(unsigned int index) const;
    CUDAORCPP bool piece(unsigned int index) const;
    CUDAORCPP bool man(unsigned int index) const;
    CUDAORCPP bool king(unsigned int index) const;

    // Move a piece from dst to src and remove the piece at capture.
    // All indices are between 0-63 
    CUDAORCPP void captureMove(unsigned int dst, unsigned int capture, unsigned int src);
    
    // Move a piece from dst to src.
    // if DEBUG is defined, it will complain if dst is occupied by
    // a piece that is owned by the player that is moving
    CUDAORCPP void move(unsigned int dst, unsigned int src);

    // Returns true if there is a man at src, and dst is empty in the last row
    // for the piece in question at src
    CUDAORCPP bool moveWillKing(unsigned int dst, unsigned int src) const;

    // Check if a piece can move to a particular index location
    // indexes should be 0-63.
    // p1 should be true for player1's move, false otherwise
    CUDAORCPP bool canMove(unsigned int dst, unsigned int src, bool p1) const;

    // Returns whether an index 0-63 is a space that can be occupied.
    CUDAORCPP bool isOccupiableSpace(unsigned int index) const;
   
    // Translate an index in 0-63 that points to an occupiable space into
    // an index 0-31 for our register operations.
    CUDAORCPP unsigned int indexToOccupiableIndex(unsigned int index) const;

    // Add a man or a king for a player to a particular index of the board
    // the index should be 0-63.
    CUDAORCPP void addP1Man(unsigned int index);
    CUDAORCPP void addP2Man(unsigned int index);
    CUDAORCPP void addP1King(unsigned int index);
    CUDAORCPP void addP2King(unsigned int index);

    // Swap the game position of the two players.
    CUDAORCPP void swapPlayers(void);
    CUDAORCPP bool getP1Up();
    CUDAORCPP void setP1Up(bool p1IsUp);
    CUDAORCPP void clear();
    
private:
    // Return the number of high bits in a register.
    CUDAORCPP size_t count(uint32_t reg) const;
    
    // Reverse the bits in a register
    CUDAORCPP void rev(uint32_t& reg);

    CUDAORCPP char getPieceChar(unsigned int index) const;

    // Set a particular bit in a register. Index should be 0-31.
    // Return the previous value of that bit
    // these functions do not check or care whether reg[index] is set or not
    CUDAORCPP bool set(uint32_t& reg, unsigned int index, bool setVal);

/* storing the board

    Only the 32 black squares on the board can ever be occupied. One register 
    will mark if the piece is a king, and the other two will mark whether a
    piece is belongs to player 1 or player 2.

    Only certain indices can ever be occupied.

    If we number the board like (1), with the starting positions in (2),
    condensed to only black squares in (3)

       (1)          (2)        (3)

    0.......  |  .0.0.0.0 | 00000000 | 
    8.......  |  0.0.0.0. | 0000.... |
    16......  |  .0.0.0.0 | ....xxxx |
    24......  |  ........ | xxxxxxxx |
    32......  |  ........ |          |
    40......  |  x.x.x.x. |          |
    48......  |  .x.x.x.x |          |
    56....63  |  x.x.x.x. |          |

    Therefore, at the beginning of the game, we will start with these values:
      p1   = 0x00000FFF
      p2   = 0xFFF00000
      king = 0x00000000
*/

    uint32_t p1;
    uint32_t p2;
    uint32_t kings;
    bool p1Up;
};


/* 
   CHECKERSBOARD.CPP 
   Implementation
*/

using namespace std;

CUDAORCPP CheckersBoard::CheckersBoard()
 : p1(0x00000FFF), p2(0xFFF00000), 
   kings(0x00000000),p1Up(true)
{
    // Nothing to do but set up the initial board state.
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(count(p1) == 12);
    assert(count(p2) == 12);
    assert(count(kings) == 0);
    assert(set(p1, 0, false));
    assert(!set(p1, 0, true));
    assert(set(p1, 0, true));
    assert(set(p1, 1, true));
    assert(set(p1, 2, true));
    assert(set(p1, 3, true));
    assert(set(p1, 4, true));
    assert(set(p1, 5, true));
    assert(set(p1, 6, true));
    assert(set(p1, 7, true));
    assert(set(p1, 8, true));
    assert(set(p1, 9, true));
    assert(set(p1, 10, true));
    assert(set(p1, 11, true));
    assert(set(p2, 20, true));
    assert(set(p2, 21, true));
    assert(set(p2, 22, true));
    assert(set(p2, 23, true));
    assert(set(p2, 24, true));
    assert(set(p2, 25, true));
    assert(set(p2, 26, true));
    assert(set(p2, 27, true));
    assert(set(p2, 28, true));
    assert(set(p2, 29, true));
    assert(set(p2, 30, true));
    assert(set(p2, 31, true));

#endif
}

CUDAORCPP CheckersBoard::CheckersBoard(uint32_t p1reg, uint32_t p2reg, uint32_t kingsreg)
 : p1(p1reg), p2(p2reg), kings(kingsreg), p1Up(true)
{
    // Nothing else to do.
}



CUDAORCPP bool CheckersBoard::operator==(const CheckersBoard& rhs) const
{
    return (p1 == rhs.p1) && (p2==rhs.p2) && (kings == rhs.kings) && (p1Up == rhs.p1Up);
}


CUDAORCPP size_t CheckersBoard::count(uint32_t reg) const
{
    // clang++ -O3, results in 20 instructions and no branches
    uint32_t n;

    n = (reg >> 1) & 0x77777777;
    reg = reg - n;
    n = (n >> 1)   & 0x77777777;
    reg = reg - n;
    n = (n >> 1)   & 0x77777777;
    reg = reg - n;
    reg = (reg + (reg >> 4)) & 0x0F0F0F0F;
    reg = reg * 0x01010101;
    return reg >> 24;
}



CUDAORCPP void CheckersBoard::rev(uint32_t& reg)
{
    reg = ((reg & 0x55555555) <<  1) | ((reg & 0xAAAAAAAA) >>  1);
    reg = ((reg & 0x33333333) <<  2) | ((reg & 0xCCCCCCCC) >>  2);
    reg = ((reg & 0x0F0F0F0F) <<  4) | ((reg & 0xF0F0F0F0) >>  4);
    reg = ((reg & 0x00FF00FF) <<  8) | ((reg & 0xFF00FF00) >>  8);
    reg = ((reg & 0x0000FFFF) << 16) | ((reg & 0xFFFF0000) >> 16);
}


void CheckersBoard::dump(std::ostream & o) const
{
    for (size_t i = 0; i < 64; ++i)
    {
        if ((i % 8 == 0) && (i != 0) ) o << "\n";

        if (isOccupiableSpace(i)) o << getPieceChar(i);
        else o << "_";
    }
    o << endl;
}

CUDAORCPP void CheckersBoard::toString(char* boardString) //board string must be size [9*8+1]
{
    
    int i = 0;
    for (int pieceIndex = 0; pieceIndex < 64; ++i,++pieceIndex)
    {
        if ((pieceIndex % 8 == 0) && (pieceIndex != 0) )
        {
            boardString[i] = '\n';
            i++;
        }
        if (isOccupiableSpace(pieceIndex)) boardString[i] = getPieceChar(pieceIndex);
        else boardString[i] = '_';
    }
    boardString[i] = '\0';
}



CUDAORCPP size_t CheckersBoard::p1NumPieces() const
{
    size_t n = count(p1);
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    // Shouldn't ever be larger than 12.
    if (n > 12) {dump(cout);}
    assert(n <= 12  && "p1 has more than 12 pieces!");
#endif
    return n;
}

CUDAORCPP size_t CheckersBoard::p1NumKings() const
{
    size_t n = count(p1 & kings);
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    // Shouldn't ever be larger than 12.
    if (n > 12) {dump(cout);}
    assert(n <= 12);
#endif
    return n;
}



CUDAORCPP size_t CheckersBoard::p2NumPieces() const
{
    size_t n = count(p2);
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    if (n > 12) {dump(cout);}
    assert(n <= 12 && "p2 has more than 12 pieces.");
#endif
    return n;
}


CUDAORCPP size_t CheckersBoard::p1NumMen() const
{
    size_t n = count(p1 & ~kings);
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    // Shouldn't ever be larger than 12.
    if (n > 12) {dump(cout);}
    assert(n <= 12);
#endif
    return n;
}


CUDAORCPP size_t CheckersBoard::p2NumMen() const
{
    size_t n = count(p2 & ~kings);
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    // Shouldn't ever be larger than 12.
    if (n > 12) {dump(cout);}
    assert(n <= 12);
#endif
    return n;
}


CUDAORCPP size_t CheckersBoard::p2NumKings() const
{
    size_t n = count(p2 & kings);
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    // Shouldn't ever be larger than 12.
    if (n > 12) {dump(cout);}
    assert(n <= 12);
#endif
    return n;
}



CUDAORCPP size_t CheckersBoard::p1WeightedMen() const
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    size_t n = count(p1);
    // Shouldn't ever be larger than 12.
    if (n > 12) {dump(cout);}
    assert(n <= 12);
#endif
    size_t val = 0;
    for (uint8_t i = 0; i < 64; ++i) {
        if (!isOccupiableSpace(i)) continue;
        if (p1Man(i)) val += 5 + i/8;
    }
    return val;
}


CUDAORCPP size_t CheckersBoard::p2WeightedMen() const
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    size_t n = count(p2);
    // Shouldn't ever be larger than 12.
    if (n > 12) {dump(cout);}
    assert(n <= 12);
#endif
    size_t val = 0;
    for (uint8_t i = 0; i < 64; ++i) {
        if (!isOccupiableSpace(i)) continue;
        if (p2Man(i)) val += 5 + (63-i)/8;
    }
    return val;
}



CUDAORCPP bool CheckersBoard::p1Man(unsigned int index) const
{
    return p1Piece(index) && !p1King(index) && (index < 64);
}



CUDAORCPP bool CheckersBoard::p2Man(unsigned int index) const
{
    return p2Piece(index) && !p2King(index) && (index < 64);
}



CUDAORCPP bool CheckersBoard::p1King(unsigned int index) const
{
    unsigned int regIndex = indexToOccupiableIndex(index);
    return p1Piece(index) && bool((kings >> regIndex) & 1);
}



CUDAORCPP bool CheckersBoard::p2King(unsigned int index) const
{
  unsigned int regIndex = indexToOccupiableIndex(index);
  return p2Piece(index) && bool((kings >> regIndex) & 1);
}


// FIXME: possible optimization target if inlining doesn't work
CUDAORCPP bool CheckersBoard::piece(unsigned int index) const
{
  return p1Piece(index) || p2Piece(index);
}



CUDAORCPP bool CheckersBoard::man(unsigned int index) const
{
  return p1Man(index) || p2Man(index);
}


// FIXME: optimize to look at just king field instead of two calls.
CUDAORCPP bool CheckersBoard::king(unsigned int index) const
{
  return p1King(index) || p2King(index);
}

CUDAORCPP void CheckersBoard::captureMove(unsigned int dst, unsigned int capture, unsigned int src)
{
#if defined (DEBUG) && !defined (__CUDA_ARCH__)
    assert(isOccupiableSpace(dst));
    assert(isOccupiableSpace(capture));
    assert(isOccupiableSpace(src));
#endif

   unsigned int regCapture = indexToOccupiableIndex(capture);
   set(kings, regCapture, false);
   set(p1, regCapture, false);
   set(p2, regCapture, false);
   move(dst, src);

}

CUDAORCPP void CheckersBoard::move(unsigned int dst, unsigned int src)
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    // make sure dst is an occupiable space
    assert(isOccupiableSpace(dst));
    // make sure src is an occupiable space
    assert(isOccupiableSpace(src));
    // Make sure there actually is a piece in the source.
    assert(p1Piece(src) || p2Piece(src));
    // Make sure both players don't claim to have a piece there.
    assert(!(p1Piece(src) && p2Piece(src)));
    // Make sure you're not moving onto another piece
    assert(!p1Piece(dst));
    assert(!p2Piece(dst));
#endif

    unsigned int regDst = indexToOccupiableIndex(dst);
    unsigned int regSrc = indexToOccupiableIndex(src);

    // Move the king register
    bool oldKing = set(kings, regSrc, false);
    set(kings, regDst, oldKing);

    // Move the piece register
    // it's safe to move both player registers since neither should have any
    // overlapping set bits.
    bool oldP1 = set(p1, regSrc, false);
    bool oldP2 = set(p2, regSrc, false);
    set(p1, regDst, oldP1);
    set(p2, regDst, oldP2);

    // Upgrade the piece to a king if it moves into the last row
    uint8_t dstRow = dst / 8;
    if (dstRow == 0 || dstRow == 7)
    {
        set(kings, regDst, true);
    }
}

CUDAORCPP bool CheckersBoard::moveWillKing(unsigned int dst, unsigned int src)
    const
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(isOccupiableSpace(dst));
    assert(isOccupiableSpace(src));
#endif
  bool emptyDst = !p1Piece(dst) && !p2Piece(dst);
  if (p1Man(src) && emptyDst) return (dst >= 56);
  else if (p2Man(src) && emptyDst) return (dst < 8);
  else return false;
}

CUDAORCPP bool CheckersBoard::canMove(unsigned int dst, unsigned int src, bool p1Turn) const
{

    // About 2k assembly instructions at -03.

#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    // make sure dst is an occupiable space
    assert(isOccupiableSpace(dst));
    // make sure src is an occupiable space
    assert(isOccupiableSpace(src));
#endif

    bool good = dst<64 && src<64;
    good = good && ((p1Piece(src) && p1Turn) || (p2Piece(src) && !p1Turn));
    good = good && !(p1Piece(dst) || p2Piece(dst)); // can't have piece at the dst

    const uint8_t srcRow = src / 8;
    const uint8_t srcCol = src % 8;
    const uint8_t dstRow = dst / 8;
    const uint8_t dstCol = dst % 8;

    if (p1Turn)
    {
        good = good && ( ((srcRow-1 == dstRow) && (srcCol+1 == dstCol) && p1King(src)                               ) ||
                         ((srcRow-1 == dstRow) && (srcCol-1 == dstCol) && p1King(src)                               ) ||
                         ((srcRow+1 == dstRow) && (srcCol+1 == dstCol)                                              ) || 
                         ((srcRow+1 == dstRow) && (srcCol-1 == dstCol)                                              ) ||
                         ((srcRow-2 == dstRow) && (srcCol+2 == dstCol) && p2Piece(srcRow-1, srcCol+1) && p1King(src)) ||
                         ((srcRow-2 == dstRow) && (srcCol-2 == dstCol) && p2Piece(srcRow-1, srcCol-1) && p1King(src)) ||
                         ((srcRow+2 == dstRow) && (srcCol+2 == dstCol) && p2Piece(srcRow+1, srcCol+1)               ) ||
                         ((srcRow+2 == dstRow) && (srcCol-2 == dstCol) && p2Piece(srcRow+1, srcCol-1)               )    );
    } 
    else
    {
        good = good && ( ((srcRow-1 == dstRow) && (srcCol+1 == dstCol)                                              ) ||
                         ((srcRow-1 == dstRow) && (srcCol-1 == dstCol)                                              ) ||
                         ((srcRow+1 == dstRow) && (srcCol+1 == dstCol) && p2King(src)                               ) || 
                         ((srcRow+1 == dstRow) && (srcCol-1 == dstCol) && p2King(src)                               ) ||
                         ((srcRow-2 == dstRow) && (srcCol+2 == dstCol) && p1Piece(srcRow-1, srcCol+1)               ) ||
                         ((srcRow-2 == dstRow) && (srcCol-2 == dstCol) && p1Piece(srcRow-1, srcCol-1)               ) ||
                         ((srcRow+2 == dstRow) && (srcCol+2 == dstCol) && p1Piece(srcRow+1, srcCol+1) && p2King(src)) ||
                         ((srcRow+2 == dstRow) && (srcCol-2 == dstCol) && p1Piece(srcRow+1, srcCol-1) && p2King(src))    );
    }

    return good;
}

CUDAORCPP bool CheckersBoard::isOccupiableSpace(unsigned int index) const
{
    unsigned int row = index / 8;
    unsigned int col = index % 8; 
    return (((row % 2 == 0) & (col % 2 == 1)) | ((row % 2 == 1) & (col % 2 == 0))) && (index < 64);
}



CUDAORCPP unsigned int CheckersBoard::indexToOccupiableIndex(unsigned int index) const
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(isOccupiableSpace(index));
#endif
    // Determine our row and column
    unsigned int row = index / 8;
    unsigned int col = index % 8;

    // Four occupiable spaces per row,
    // every other column is occupiable.
    unsigned int occupiableIndex = row * 4 + (col / 2);
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(occupiableIndex < 32);
#endif
    return occupiableIndex;
}



CUDAORCPP void CheckersBoard::addP1Man(unsigned int index)
{
    set(p1, indexToOccupiableIndex(index), 1);
}



CUDAORCPP void CheckersBoard::addP2Man(unsigned int index)
{
    set(p2, indexToOccupiableIndex(index), 1);
}


CUDAORCPP void CheckersBoard::addP1King(unsigned int index)
{
    const unsigned int regIndex = indexToOccupiableIndex(index);
    set(p1, regIndex, 1);
    set(kings, regIndex, 1);
}


CUDAORCPP void CheckersBoard::addP2King(unsigned int index)
{
    const unsigned int regIndex = indexToOccupiableIndex(index);
    set(p2, regIndex, 1);
    set(kings, regIndex, 1);
}

CUDAORCPP void CheckersBoard::swapPlayers()
{
    rev(p1);
    rev(p2);
    rev(kings);
    unsigned int temp = p1;
    p1 = p2;
    p2 = temp;
    p1Up = !p1Up;
}

CUDAORCPP bool CheckersBoard::getP1Up()
{
   return p1Up;
}

CUDAORCPP void CheckersBoard::setP1Up(bool p1IsUp)
{
    p1Up = p1IsUp;
}
CUDAORCPP void CheckersBoard::clear()
{
   kings = p2 = p1 = 0;
}


CUDAORCPP bool CheckersBoard::p1Piece(unsigned int index) const
{
    unsigned int regIndex = indexToOccupiableIndex(index);
    return isOccupiableSpace(index) && bool((p1 >> regIndex) & 1);
}



CUDAORCPP bool CheckersBoard::p1Piece(uint8_t row, uint8_t col) const
{
    unsigned int index = row * 8 + col;
    unsigned int regIndex = indexToOccupiableIndex(index);
    return (row < 8) && (col < 8) && isOccupiableSpace(index) && bool((p1 >> regIndex) & 1);
}



CUDAORCPP bool CheckersBoard::p2Piece(unsigned int index) const
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(index < 64);
#endif
    unsigned int regIndex = indexToOccupiableIndex(index);
    return isOccupiableSpace(index) && bool((p2 >> regIndex) & 1);
}



CUDAORCPP bool CheckersBoard::p2Piece(uint8_t row, uint8_t col) const
{
    unsigned int index = row * 8 + col;
    unsigned int regIndex = indexToOccupiableIndex(index);
    return (row < 8) && (col < 8) && isOccupiableSpace(index) && bool((p2 >> regIndex) & 1);
}



CUDAORCPP char CheckersBoard::getPieceChar(unsigned int index) const
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(isOccupiableSpace(index));
#endif
    bool p1man, p1king, p2man, p2king;

    p1man  = p1Man(index);
    p1king = p1King(index);
    p2man  = p2Man(index);
    p2king = p2King(index);

#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(~(p1man && p1king));
    assert(~(p1man && p2man));
    assert(~(p1man && p2king));
    assert(~(p2man && p1king));
    assert(~(p2man && p2king));
    assert(~(p1king && p2king));
#endif

    if      (p1man)  return 'o';
    else if (p1king) return 'O';
    else if (p2man)  return 't';
    else if (p2king) return 'T';
    else             return '.';
}

CUDAORCPP bool CheckersBoard::set(uint32_t& reg, unsigned int index, bool setVal)
{
#if defined(DEBUG) && !defined(__CUDA_ARCH__)
    assert(index < 32);
#endif
    // Keep track of the old value to return
    bool oldVal = (reg >> index) & 1;
    
    if (setVal) reg |=               (uint32_t(1) << index);
    else        reg &= (0xFFFFFFFF - (uint32_t(1) << index));

    return oldVal;
}

#endif
