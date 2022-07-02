DEBUGDEFS = -DDEBUG

# ----- Make Macros -----

CXXFLAGS = -Wall -Wextra -Wshadow -pedantic -O3
CXX     = g++ $(DEBUGDEFS) -g

TARGETS = main 
OBJECTS = main.o checkersboard.o CheckersSearch.o PerformanceAnalyzer.o HumanPlayer.o


# ----- Make Rules -----

all:    $(TARGETS)

main:   $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o main $(OBJECTS) $(SDL)

clean:
	rm -f $(TARGETS) $(OBJECTS)

# ----8<----8<----8<----8<----8<----8<----8<----8<----8<----8<----8<----8<----
checkersboard.o: checkersboard.cpp
CheckersSearch.o: CheckersSearch.cpp CheckersSearch.hpp checkersboard.cpp
main.o: main.cpp checkersboard.cpp CheckersSearch.hpp
CheckersSearch.o: CheckersSearch.hpp checkersboard.cpp
