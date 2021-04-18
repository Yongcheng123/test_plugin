all: main.o atom.o mrc.o
	g++ -o bin/visualization.out obj/main.o obj/atom.o obj/mrc.o

main.o:
	g++ -c src/main.cpp -o obj/main.o

atom.o:
	g++ -c src/ATOM.cpp -o obj/atom.o

mrc.o:
	g++ -c src/MRC.cpp -o obj/mrc.o
