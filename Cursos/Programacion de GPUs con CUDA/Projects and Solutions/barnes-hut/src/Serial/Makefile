CC=gcc
CFLAGS=-c -Wall -std=c99
LDFLAGS= -lm
SOURCES=nbody.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=nbody

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE)
