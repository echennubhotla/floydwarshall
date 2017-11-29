 /*Author: Esha Chennubhotla
 *File:     dijkstra.c
 * Purpose:  Implement Dijkstra's algorithm for solving the single-source 
 *           shortest path problem:  find the length of the shortest path 
 *           between a specified vertex and all other vertices in a 
 *           directed graph.
 *
 * Input:    n, the number of vertices in the digraph
 *           mat, the adjacency matrix of the digraph
 * Output:   A list showing the cost of the shortest path
 *           from vertex 0 to every other vertex in the graph.
 *           and the vertices on the shortest path from 0 to
 *           every other vertex.
 *
 * Compile: mpicc -g -Wall -o dijkstra dijsktra.c
 * Run:     mpiexec -n <number of processes> dijkstra
 *           For large matrices, put the matrix into a file with n as
 *           the first line and run with ./dijkstra < large_matrix
 *
 * Notes:
 * 1.  Edge lengths should be nonnegative.
 * 2.  The distance from v to w may not be the same as the distance from
 *     w to v.
 * 3.  If there is no edge between two vertices, the length is the constant
 *     INFINITY.  So input edge length should be substantially less than
 *     this constant.
 * 4.  The cost of travelling from a vertex to itself is 0.  So the adjacency
 *     matrix has zeroes on the main diagonal.
 * 5.  Edges joining different vertices have positive weights.
 * 6.  No error checking is done on the input.
 * 7.  The adjacency matrix is stored as a 1-dimensional array and subscripts
 *     are computed using the formula:  the entry in the ith row and jth
 *     column is mat[i*n + j]
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>


const int INFINITY = 1000000;

int Read_matrix(int* mat, int n);
void Print_dists(int* dist, int n);
void Print_matrix(int* mat, int* loc_mat, int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
int  Find_min_dist(int* loc_dist, int* known, int loc_n, int my_rank, MPI_Comm comm);
void Dijkstra(int* loc_mat, int loc_n, int my_rank, int n, int* loc_dist, int* loc_pred, int* known, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Print_paths(int* pred, int* path, int n);


int main(int argc, char* argv[]) {
   MPI_Comm comm;
   MPI_Datatype blk_col_mpi_t = NULL;
   int p, my_rank;
   int  n = 0, loc_n = 0;
   int *loc_mat = NULL, *known = NULL;
   int *loc_dist = NULL, *loc_pred= NULL;
   int *mat = NULL;
   int *dist = NULL;
   int *pred = NULL;
   int *path = NULL;

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &my_rank);

   if (my_rank ==0){
      printf("How many vertices?\n");
      scanf("%d", &n);
      mat = malloc(n*n*sizeof(int));
      dist = malloc(n*sizeof(int));
      pred = malloc(n*sizeof(int));
      path =  malloc(n*n*sizeof(int));
      Read_matrix(mat, n);
   } 

   MPI_Bcast(&n, 1, MPI_INT, 0, comm);

   loc_n = n/p;
   known = malloc(loc_n*sizeof(int));
   loc_mat = malloc(n*loc_n*sizeof(int));
   loc_dist = malloc(loc_n*sizeof(int));
   loc_pred = malloc(loc_n*sizeof(int));
   blk_col_mpi_t = Build_blk_col_type(n, loc_n);


   MPI_Scatter(mat, 1, blk_col_mpi_t,loc_mat, n*loc_n, MPI_INT, 0, comm);
   
   Dijkstra(loc_mat, loc_n, my_rank, n, loc_dist, loc_pred, known, comm);

   MPI_Gather(loc_dist, loc_n, MPI_INT, dist, 1, blk_col_mpi_t, 0, comm);
   MPI_Gather(loc_pred, loc_n, MPI_INT, pred, 1, blk_col_mpi_t, 0, comm);

   if (my_rank == 0){
      Print_dists(dist, n);
      Print_paths(pred, path, n);
      free(mat);
      free(dist);
      free(pred);
      free(path);
   }

   free(loc_mat);
   free(loc_pred);
   free(loc_dist);
   free(known);
   MPI_Type_free(&blk_col_mpi_t);

   MPI_Finalize();
   return 0;
}  /* main */

/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n) {
   MPI_Aint lb, extent;
   MPI_Datatype block_mpi_t;
   MPI_Datatype first_bc_mpi_t;
   MPI_Datatype blk_col_mpi_t;

   MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
   MPI_Type_get_extent(block_mpi_t, &lb, &extent);

   MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
   MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
         &blk_col_mpi_t);
   MPI_Type_commit(&blk_col_mpi_t);

   MPI_Type_free(&block_mpi_t);
   MPI_Type_free(&first_bc_mpi_t);

   return blk_col_mpi_t;
}  /* Build_blk_col_type */

/*---------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in an nxn matrix of ints on process 0, and
 *            distribute it among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 * In args:   n:  the number of rows in the matrix and the submatrices
 *            loc_n = n/p:  the number of columns in the submatrices
 *            blk_col_mpi_t:  the MPI_Datatype used on process 0
 *            my_rank:  the caller's rank in comm
 *            comm:  Communicator consisting of all the processes
 * Out arg:   loc_mat:  the calling process' submatrix (needs to be 
 *               allocated by the caller)
 */
int Read_matrix(int* mat, int n){
   int i, j;

   printf("Enter your matrix:\n");
   for(i = 0; i < n; i++){
      for (j = 0; j < n; j++){
         scanf(" %d", &mat[i*n + j]);
      }
   }

   return *mat;
} 

/*-------------------------------------------------------------------
 * Function:    Dijkstra
 * Purpose:     Apply Dijkstra's algorithm to the matrix mat
 * In args:     n:  the number of vertices
 *              mat:  adjacency matrix for the graph
 * Out args:    dist:  dist[v] = distance 0 to v.
 *              pred:  pred[v] = predecessor of v on a 
 *                  shortest path 0->v.
*/

void Dijkstra(int* loc_mat, int loc_n, int my_rank, int n, int* loc_dist, int* loc_pred, int* known, 
   MPI_Comm comm) {

   int loc_v, loc_w, loc_u, new_dist, u;
   int my_min[2], glbl_min[2];
   int dist_u;

   for (loc_v = 0; loc_v < loc_n; loc_v++) {
      loc_dist[loc_v] = loc_mat[0*loc_n+ loc_v];
      loc_pred[loc_v] = 0;
      known[loc_v] = 0;
   }

   if (my_rank == 0){
      known[0] = 1;
   }

   for (loc_v = 1; loc_v < n; loc_v++){
      loc_u = Find_min_dist(loc_dist, known, loc_n, my_rank, comm);

      if (loc_u >= 0){
         my_min[0] = loc_dist[loc_u];
         my_min[1] = loc_u + my_rank* loc_n;
      }
      else{
         my_min[0] = INFINITY;
         my_min[1] = -1;
      }


      MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);

      u = glbl_min[1];
      dist_u = glbl_min[0] ;

      if(u/loc_n == my_rank){
         loc_u = u % loc_n;
         known[loc_u] = 1;
      }

      for(loc_w = 0; loc_w < loc_n; loc_w++){
         if(!known[loc_w]){ 
            new_dist = dist_u + loc_mat[u*loc_n + loc_w];
            if (new_dist < loc_dist[loc_w]){
               loc_dist[loc_w] = new_dist;
               loc_pred[loc_w] = u;
            }
         }
      }
   }
 /* Dijkstra */
}
/*-------------------------------------------------------------------
 * Function:    Find_min_dist
 * Purpose:     Find the minimum distance of a vertex
 */
int  Find_min_dist(int* loc_dist, int* known, int loc_n, int my_rank, MPI_Comm comm){

   int min_dist, v, u;

   u = -1; 
   min_dist = INFINITY + 1;
   for(v = 0; v < loc_n; v++){
      if (!known[v]){
         if (loc_dist[v] < min_dist){
            u = v;
            min_dist = loc_dist[v];
         }
      }
   }

   return u;
}



/*-------------------------------------------------------------------
 * Function:    Print_dists
 * Purpose:     Print the length of the shortest path from 0 to each
 *              vertex
 * In args:     n:  the number of vertices
 *              dist:  distances from 0 to each vertex v:  dist[v]
 *                 is the length of the shortest path 0->v
 */
void Print_dists(int* dist, int n){
   int v;

   printf("The distance from 0 to each vertex is:\n");
   printf("  v    dist 0->v\n");
   printf("----   ---------\n");
               
   for (v = 1; v < n; v++)
      printf("%3d       %4d\n", v, dist[v]);
   printf("\n");

} /* Print_dists */  

/*-------------------------------------------------------------------
 * Function:    Print_paths
 * Purpose:     Print the shortest path from 0 to each vertex
 * In args:     n:  the number of vertices
 *              pred:  list of predecessors:  pred[v] = u if
 *                 u precedes v on the shortest path 0->v
 */
void Print_paths(int* pred, int* path, int n){
   int v, w, count, i;

   printf("The shortest path from 0 to each vertex is:\n");
   printf("  v     Path 0->v\n");
   printf("----    ---------\n");
   for (v = 1; v < n; v++) {
      printf("%3d:    ", v);
      count = 0;
      w = v;
      while (w != 0) {
         path[count] = w;
         count++;
         w = pred[w];
      }
      printf("0 ");
      for (i = count-1; i >= 0; i--)
         printf("%d ", path[i]);
      printf("\n");
   }

}  /* Print_paths */