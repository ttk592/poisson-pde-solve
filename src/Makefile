CC=gcc
CPP=g++
CFLAGS=-O2 -Wall # -DTIME_REPORT
LFLAGS=-lm -lfftw3 -lfftw3_threads

PROGS=example reg_test

all:		${PROGS}

example:	example.o laplace.o
		${CPP} example.o laplace.o ${LFLAGS} -o example

reg_test:	reg_test.o laplace.o
		${CPP} reg_test.o laplace.o ${LFLAGS} -lboost_unit_test_framework -o reg_test


%.o: %.cpp	
		${CPP} ${CFLAGS} -c $< -o $@

%.o: %.c
		${CC} ${CFLAGS} -c $< -o $@

clean:
		rm -f ${PROGS} *.o
