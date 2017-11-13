 /* Author: Esha Chennubhotla
 * File:     floydwarshall.c
 *
 * Purpose:  Parallelizes the floyd marshall alogorithm
 * 
 * Input:    Number of processes and matrix of positive ints nxn of cost of cities
 * Output:   Cheapest route to get from one city to another
 *
 * Compile:  mpicc -g -Wall -o floydmarshall floydwarshall.c
 * Run:      mpiexec -n <number of processes> floydwarshall
 *
 * Notes:     
 *    1.  nxn matrix of positive ints
 *    2.  diagonal entries in the matriz i=j will be 0
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define INFINITY 1000000
#define MAX_COST 10

void Usage(char* prog_name);
void Print_row(int* local_mat, int n, int my_rank, int i);
void Print_matrix(int* mat, int n);
void Floyd(int* local_mat, int n, int p, int my_rank);
int getMatrix(void);
int Read_Matrix(int* mat, int n);
int min(int* local_mat, int local_city1, int n, int int_city, int* row_int_city, int city2);

int main(int argc, char* argv[]) {
	MPI_Comm comm;
	int n = 0;
	int * local_mat = NULL;
	int* mat = NULL;
	int p, my_rank;

	MPI_Init(&argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);


	if (my_rank == 0) {
		printf("Enter n: ");
		fflush(stdout);
		scanf("%d", &n);
		mat = malloc(n*n*sizeof(int));
		Read_Matrix(mat, n);
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	local_mat = malloc(n*n/p*sizeof(int));
	MPI_Scatter(mat, n*n/p, MPI_INT,local_mat, n*n/p, MPI_INT, 0, comm);

	Floyd(local_mat, n ,p, my_rank);

	MPI_Gather(local_mat, n*n/p, MPI_INT, mat, n*n/p, MPI_INT, 0, comm);

	if (my_rank == 0){
		printf("The solution is:\n");
		Print_matrix(mat, n);
		free(mat);
	}

	free(local_mat);
	MPI_Finalize();
	return 0;
}


/*-------------------------------------------------------------------
 * Function:    Floyd
 * Purpose:     Apply Floyd's algorithm to the matrix mat
 * In arg:      n
 * In/out arg:  mat:  on input, the adjacency matrix, on output
 *              lengths of the shortest paths between each pair of
 *              vertices.
 */

void Floyd(int* local_mat, int n, int p, int my_rank) {
	int root = 0;
	int local_int_city = 0;
	int* row_int_city = NULL;

	row_int_city = malloc(n*sizeof(int));

	for (int int_city = 0; int_city < n; int_city++) {
		root = int_city/(n/p);
		if (my_rank == root) {
			local_int_city = int_city % (n/p);
			for (int j = 0; j < n; j++){
				row_int_city[j] = local_mat[local_int_city*n + j];
			}
		}
		MPI_Bcast(row_int_city, n, MPI_INT, root, MPI_COMM_WORLD);
		for (int local_city1 = 0; local_city1 < n/p ; local_city1++){
			for (int city2 = 0; city2 < n; city2++){
				local_mat[local_city1*n + city2] = min(local_mat, local_city1, n, int_city, row_int_city, city2);
			}
		}
	}
	free(row_int_city);
}  /* Floyd */

/*-------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print the contents of the matrix
 * In args:   mat, n
 */
void Print_matrix(int* mat, int n) {
   int i, j;

   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
         if (mat[i*n+j] == INFINITY)
            printf("i ");
         else
            printf("%d ", mat[i*n+j]);
      printf("\n");
   }
}  

void Usage(char* prog_name) {
   fprintf(stderr, "usage:  %s <number of rows>\n", prog_name);
   exit(0);
}  

/* Usage */

int Read_Matrix(int* mat, int n){
	printf("Enter your matrix:\n");
	for(int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			fflush(stdout);
			scanf("%d", &mat[i*n + j]);
		}
	}

	return *mat;

}

/*---------------------------------------------------------------------
 * Function:  Print_row
 * Purpose:   Convert a row of local_mat to a string and then print
 *            the row.  This tends to reduce corruption of output
 *            when multiple processes are printing.
 * In args:   all            
 */
int min(int* local_mat, int local_city1, int n, int int_city, int* row_int_city, int city2){

	int firstCalc;
	int secCalc;

	firstCalc = local_mat[local_city1*n + city2];
	secCalc = local_mat[local_city1*n + int_city]+ row_int_city[city2];

	if (firstCalc > secCalc){
		return secCalc;
	}
	else{
		return firstCalc;
	}
}

void Print_row(int local_mat[], int n, int my_rank, int i){
   char char_int[100];
   char char_row[1000];
   int j, offset = 0;

   for (j = 0; j < n; j++) {
      if (local_mat[i*n + j] == INFINITY)
         sprintf(char_int, "i ");
      else
         sprintf(char_int, "%d ", local_mat[i*n + j]);
      sprintf(char_row + offset, "%s", char_int);
      offset += strlen(char_int);
   }  
   printf("Proc %d > row %d = %s\n", my_rank, i, char_row);
}  /* Print_row */# floydwarshall
