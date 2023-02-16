WARNING = -Wall -Wshadow --pedantic
ERROR = -Wvla #-Werror
GCC = gcc -std=c99 -g $(WARNING) $(ERROR) 
VAL = valgrind --tool=memcheck --log-file=vallog --leak-check=full --verbose
TESTFALGS =#-DDEBUG2 #-DDEBUG 

SRCS = *.c
OBJS = $(SRCS:%.c=%.o)

main: $(OBJS)
	$(GCC) $(OBJS) -o main

.c.o: 
	$(GCC) $(TESTFALGS) -c $*.c 


clean: # remove all machine generated files
	rm -f main *.o



