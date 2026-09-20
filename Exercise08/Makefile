CC = mpicc

PROGRAMS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

all: $(PROGRAMS)

sum_bcast: sum_bcast.c
	$(CC) -o sum_bcast sum_bcast.c

sum_scatter: sum_scatter.c
	$(CC) -o sum_scatter sum_scatter.c

sum_gather: sum_gather.c
	$(CC) -o sum_gather sum_gather.c

sum_reduce: sum_reduce.c
	$(CC) -o sum_reduce sum_reduce.c

sum_allreduce: sum_allreduce.c
	$(CC) -o sum_allreduce sum_allreduce.c

sum_scan: sum_scan.c
	$(CC) -o sum_scan sum_scan.c

run: all
	mpirun -np 4 ./sum_bcast
	mpirun -np 4 ./sum_scatter
	mpirun -np 4 ./sum_gather
	mpirun -np 4 ./sum_reduce
	mpirun -np 4 ./sum_allreduce
	mpirun -np 4 ./sum_scan

clean:
	rm -f $(PROGRAMS)