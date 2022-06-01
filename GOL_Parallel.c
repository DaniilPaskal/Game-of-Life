/*
 File:    GOL_Parallel.c
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define ORDER 64 // order of cell matrix
#define DURATION 1000	// number of iterations program runs for
#define PRINT_FREQUENCY 100	// frequency with which cell matrix is printed (1 means every iteration, 10 means every 10 iterations, etc.)
#define GLIDER_TEST 0	// determines whether we are testing single glider (1 if true, 0 if false)

// Prints current state of cell matrix
void printCells(int a[ORDER][ORDER], int iteration) {
	int x, y;	// cell coordinates

	// Border and iteration number
	printf("\n");
	for (x = 0; x < ORDER; x++) {
		printf("-");
	}
	printf("\nIteration %d\n", iteration);

	// If cell is alive, print 0; otherwise print a space
	for (y = 0; y < ORDER; y++) {
		for (x = 0; x < ORDER; x++) {
			if (a[x][y] == 1) {
				printf("O");
			} else {
				printf(" ");
			}
		}
		printf("\n");
	}
}

int main(int argc, char **argv) {
	int rank;					// rank of processor
	int p;						// number of processors
	int i, j, k;				// loop counters
	int a[ORDER][ORDER];		// matrix of cells
	int dim[2], period[2];		// variables for creating cartesian topology
	int coords[2];				// processor coordinates
	int tempCoords[2];// temporary coordinates (for finding neighbouring processors)
	int destCoords[2];		// coordinates of process being communicated with
	int x, y;					// cell coordinates
	int xOffset, yOffset;	// offset of subarray's cells within greater matrix
	int neighbours;				// sum of neighbours around cell
	int subOrder;	// order of subarrays

	MPI_Status status;			// MPI status
	MPI_Request req;			// MPI request
	MPI_Comm comm;				// MPI comm
	MPI_Datatype column_mpi_t;	// MPI column datatype
	double time1, time2;		// Start and end times for computation
	FILE *f;					// file for writing state
	char filename[128];			// name of output file

	// Start up MPI
	MPI_Init(&argc, &argv);

	// Start keeping time
	time1 = MPI_Wtime();

	// Find out number of processors
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	// Find out process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Calculate order of subarrays (order of overall matrix divided by square root of p)
	subOrder = ORDER / (int) sqrt(p);

	// Create column data type
	MPI_Type_vector(subOrder, 1, subOrder, MPI_INT, &column_mpi_t);
	MPI_Type_commit(&column_mpi_t);

	// Set size of 2nd dimension of ghost point array to subarray order (if order < 4, set size to 4 to accomodate 4 corners)
	int ghostPointsSize = subOrder;
	if (ghostPointsSize < 4) {
		ghostPointsSize = 4;
	}

	int aLocal[subOrder][subOrder];	// subarray of cell matrix located at processor
	int aLocalNext[subOrder][subOrder];	// next iteration of local subarray
	int aTemp[subOrder][subOrder];	// temporary array buffer
	int ghostPoints[5][ghostPointsSize];// ghost points representing neighbouring cells in adjacent subarrays
	int tl, t, tr, l, r, bl, b, br; // ranks of neighbouring processes (top left, top, top right, left, right, bottom left, bottom, bottom right)

	// Set dimension and period of cartesian grid
	dim[0] = (int) sqrt((double) p);
	dim[1] = (int) sqrt((double) p);
	period[0] = 1;
	period[1] = 1;

	// Create cartesian grid
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, 1, &comm);

	// Get process coordinates
	MPI_Cart_coords(comm, rank, 2, coords);

	// In process 0, generate starting matrix
	if (rank == 0) {
		// Set seed for random number generator
		srand(time(NULL));

		// Populate cell matrix with random values
		for (i = 0; i < ORDER; i++) {
			for (j = 0; j < ORDER; j++) {
				if (GLIDER_TEST == 0) {
					a[i][j] = rand() % 2;
				} else {
					a[i][j] = 0;
				}
			}
		}

		// If glider test enabled, place glider in matrix
		if (GLIDER_TEST == 1) {
			a[0][1] = 1;
			a[1][2] = 1;
			a[2][0] = 1;
			a[2][1] = 1;
			a[2][2] = 1;
		}

		// Print starting matrix
		printCells(a, 0);

		// Loop through processes
		for (i = 1; i < p; i++) {
			// Get x and y offsets of process subarrays within matrix
			MPI_Cart_coords(comm, i, 2, destCoords);
			xOffset = destCoords[1] * subOrder;
			yOffset = destCoords[0] * subOrder;

			// Loop through section of matrix corresponding to process subarray and place values in temporary array
			for (x = 0; x < subOrder; x++) {
				for (y = 0; y < subOrder; y++) {
					aTemp[x][y] = a[x + xOffset][y + yOffset];
				}
			}

			// Send array to process
			MPI_Send(aTemp, pow(subOrder, 2), MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		// Set process 0's subarray
		for (x = 0; x < subOrder; x++) {
			for (y = 0; y < subOrder; y++) {
				aLocal[x][y] = a[x][y];
			}
		}
	} else {
		// If not process 0, receive subarray
		MPI_Recv(aLocal, pow(ORDER, 2) / p, MPI_INT, 0, 0, MPI_COMM_WORLD,
				&status);
	}

	// Get ranks of neighbouring subarrays using coordinates
	tempCoords[0] = coords[0] - 1;
	tempCoords[1] = coords[1] - 1;
	MPI_Cart_rank(comm, tempCoords, &tl);
	tempCoords[0] = coords[0] - 1;
	tempCoords[1] = coords[1];
	MPI_Cart_rank(comm, tempCoords, &t);
	tempCoords[0] = coords[0] - 1;
	tempCoords[1] = coords[1] + 1;
	MPI_Cart_rank(comm, tempCoords, &tr);
	tempCoords[0] = coords[0];
	tempCoords[1] = coords[1] - 1;
	MPI_Cart_rank(comm, tempCoords, &l);
	tempCoords[0] = coords[0];
	tempCoords[1] = coords[1] + 1;
	MPI_Cart_rank(comm, tempCoords, &r);
	tempCoords[0] = coords[0] + 1;
	tempCoords[1] = coords[1] - 1;
	MPI_Cart_rank(comm, tempCoords, &bl);
	tempCoords[0] = coords[0] + 1;
	tempCoords[1] = coords[1];
	MPI_Cart_rank(comm, tempCoords, &b);
	tempCoords[0] = coords[0] + 1;
	tempCoords[1] = coords[1] + 1;
	MPI_Cart_rank(comm, tempCoords, &br);

	// Loop for specified number of iterations
	for (i = 1; i < DURATION + 1; i++) {
		// Order of sendrecv calls for corners and left/right depends on rank's parity so that processes are paired with the correct neighbour
		// (e.x. if using 16 processors, processor 1 will send processor 2 own right to be 2's left as 2 sends left to be 1's right)
		if (rank % 2 == 0) {
			// Exchange corners
			MPI_Sendrecv(&(aLocal[subOrder - 1][subOrder - 1]), 1, MPI_INT, tl,
					2, &(ghostPoints[4][0]), 1, MPI_INT, tl, 2, MPI_COMM_WORLD,
					&status);
			MPI_Sendrecv(&(aLocal[0][subOrder - 1]), 1, MPI_INT, tr, 3,
					&(ghostPoints[4][1]), 1, MPI_INT, tr, 3, MPI_COMM_WORLD,
					&status);
			MPI_Sendrecv(&(aLocal[subOrder - 1][0]), 1, MPI_INT, bl, 2,
					&(ghostPoints[4][2]), 1, MPI_INT, bl, 2, MPI_COMM_WORLD,
					&status);
			MPI_Sendrecv(&(aLocal[0][0]), 1, MPI_INT, br, 3,
					&(ghostPoints[4][3]), 1, MPI_INT, br, 3, MPI_COMM_WORLD,
					&status);

			// Exchange columns
			MPI_Sendrecv(&(aLocal[subOrder - 1][0]), subOrder, MPI_INT, l, 6,
					&(ghostPoints[1][0]), subOrder, MPI_INT, l, 6,
					MPI_COMM_WORLD, &status);
			MPI_Sendrecv(&(aLocal[0][0]), subOrder, MPI_INT, r, 7,
					&(ghostPoints[2][0]), subOrder, MPI_INT, r, 7,
					MPI_COMM_WORLD, &status);
		} else {
			MPI_Sendrecv(&(aLocal[0][0]), 1, MPI_INT, br, 2,
					&(ghostPoints[4][3]), 1, MPI_INT, br, 2, MPI_COMM_WORLD,
					&status);
			MPI_Sendrecv(&(aLocal[subOrder - 1][0]), 1, MPI_INT, bl, 3,
					&(ghostPoints[4][2]), 1, MPI_INT, bl, 3, MPI_COMM_WORLD,
					&status);
			MPI_Sendrecv(&(aLocal[0][subOrder - 1]), 1, MPI_INT, tr, 2,
					&(ghostPoints[4][1]), 1, MPI_INT, tr, 2, MPI_COMM_WORLD,
					&status);
			MPI_Sendrecv(&(aLocal[subOrder - 1][subOrder - 1]), 1, MPI_INT, tl,
					3, &(ghostPoints[4][0]), 1, MPI_INT, tl, 3, MPI_COMM_WORLD,
					&status);

			MPI_Sendrecv(&(aLocal[0][0]), subOrder, MPI_INT, r, 6,
					&(ghostPoints[2][0]), subOrder, MPI_INT, r, 6,
					MPI_COMM_WORLD, &status);
			MPI_Sendrecv(&(aLocal[subOrder - 1][0]), subOrder, MPI_INT, l, 7,
					&(ghostPoints[1][0]), subOrder, MPI_INT, l, 7,
					MPI_COMM_WORLD, &status);
		}

		// Order of sendrecv calls for top/bottom depends on party of processor's row in cartesian topology
		if (coords[0] % 2 == 0) {
			// Exchange rows
			MPI_Sendrecv(&(aLocal[0][subOrder - 1]), 1, column_mpi_t, t, 8,
					&(ghostPoints[0][0]), subOrder, MPI_INT, t, 8,
					MPI_COMM_WORLD, &status);
			MPI_Sendrecv(&(aLocal[0][0]), 1, column_mpi_t, b, 9,
					&(ghostPoints[3][0]), subOrder, MPI_INT, b, 9,
					MPI_COMM_WORLD, &status);
		} else {
			MPI_Sendrecv(&(aLocal[0][0]), 1, column_mpi_t, b, 8,
					&(ghostPoints[3][0]), subOrder, MPI_INT, b, 8,
					MPI_COMM_WORLD, &status);
			MPI_Sendrecv(&(aLocal[0][subOrder - 1]), 1, column_mpi_t, t, 9,
					&(ghostPoints[0][0]), subOrder, MPI_INT, t, 9,
					MPI_COMM_WORLD, &status);
		}

		// Loop through cells in process subarray and count each cell's neighbours
		for (x = 0; x < subOrder; x++) {
			for (y = 0; y < subOrder; y++) {
				neighbours = 0;

				// Sum neighbours in row above cell
				if (y == 0) {
					// If cell is top row...
					if (x == 0) {
						// If cell is in top left corner, count top left ghost point from process to upper left
						neighbours += ghostPoints[4][0];
					} else {
						// If cell is not in left corner, count top left ghost point from process above
						neighbours += ghostPoints[0][x - 1];
					}
					// Count top ghost point from process above
					neighbours += ghostPoints[0][x];
					if (x == subOrder - 1) {
						// If cell is in top right corner, count top right ghost point from process to upper right
						neighbours += ghostPoints[4][1];
					} else {
						// If cell is not in right corner, count top right ghost point from process above
						neighbours += ghostPoints[0][x + 1];
					}
				} else {
					// If cell is not top row...
					if (x == 0) {
						// If cell is in left column, count top left ghost point from process to left
						neighbours += ghostPoints[1][y - 1];
					} else {
						// If cell is not in left column, count top left point in this process
						neighbours += aLocal[x - 1][y - 1];
					}
					// Count top point in this process
					neighbours += aLocal[x][y - 1];
					if (x == subOrder - 1) {
						// If cell is in right column, count top right ghost point from process to right
						neighbours += ghostPoints[2][y - 1];
					} else {
						// If cell is not in right column, count top right point in this process
						neighbours += aLocal[x + 1][y - 1];
					}
				}

				// Sum neighbours to left and right
				if (x == 0) {
					// If cell is in right column, count right ghost point from process to right
					neighbours += ghostPoints[1][y];
				} else {
					// If cell is not in right column, count right point in this process
					neighbours += aLocal[x - 1][y];
				}
				if (x == subOrder - 1) {
					// If cell is in left column, count ghost point from process to the left
					neighbours += ghostPoints[2][y];
				} else {
					// If cell is not in left column, count left point in this process
					neighbours += aLocal[x + 1][y];
				}

				// Sum neighbours in row below cell
				if (y == subOrder - 1) {
					// If cell is bottom row...
					if (x == 0) {
						// If cell is in bottom left corner, count bottom left ghost point from process to lower left
						neighbours += ghostPoints[4][2];
					} else {
						// If cell is not in left corner, count bottom left ghost point from process below
						neighbours += ghostPoints[3][x - 1];
					}
					// Count bottom ghost point from process below
					neighbours += ghostPoints[3][x];
					if (x == subOrder - 1) {
						// If cell is in bottom right corner, count bottom right ghost point from process to lower right
						neighbours += ghostPoints[4][3];
					} else {
						// If cell is not in right corner, count top right ghost point from process above
						neighbours += ghostPoints[3][x + 1];
					}
				} else {
					// If cell is not bottom row...
					if (x == 0) {
						// If cell is in left column, count bottom left ghost point from process to left
						neighbours += ghostPoints[1][y + 1];
					} else {
						// If cell is not in left column, count bottom left point in this process
						neighbours += aLocal[x - 1][y + 1];
					}
					// Count bottom point in this process
					neighbours += aLocal[x][y + 1];
					if (x == subOrder - 1) {
						// If cell is in right column, count bottom right ghost point from process to right
						neighbours += ghostPoints[2][y + 1];
					} else {
						// If cell is not in right column, count bottom right point in this process
						neighbours += aLocal[x + 1][y + 1];
					}
				}

				// Determine state of cell in next iteration based on neighbour sum
				if (aLocal[x][y] == 1) {
					// If cell is alive and has 2 or 3 neighbours, stay alive; otherwise die
					if (neighbours == 2 || neighbours == 3) {
						aLocalNext[x][y] = 1;
					} else {
						aLocalNext[x][y] = 0;
					}
				} else {
					// If cell is dead and has 3 neighbours, become alive; otherwise stay dead
					if (neighbours == 3) {
						aLocalNext[x][y] = 1;
					} else {
						aLocalNext[x][y] = 0;
					}
				}

				if (rank == 1) {
					printf("%d,%d has %d neighs\n", x, y, neighbours);
					/*
					 for (j = 0; j < subOrder; j++) {
					 printf("%d, %d, %d, %d\n", ghostPoints[0][j],
					 ghostPoints[1][j], ghostPoints[2][j],
					 ghostPoints[3][j]);
					 }
					 printf("%d, %d, %d, %d\n", ghostPoints[4][0], ghostPoints[4][1],
					 ghostPoints[4][2], ghostPoints[4][3]);
					 */
				}
			}
		}

		// Set local array to next array
		for (x = 0; x < subOrder; x++) {
			for (y = 0; y < subOrder; y++) {
				aLocal[x][y] = aLocalNext[x][y];
			}
		}

		// If it is time print, print matrix
		if (i % PRINT_FREQUENCY == 0) {
			// Rank 0 receives subarrays from processes, places values back in greater matrix, and prints
			if (rank == 0) {
				for (j = 1; j < p; j++) {
					MPI_Cart_coords(comm, j, 2, destCoords);
					xOffset = destCoords[1] * subOrder;
					yOffset = destCoords[0] * subOrder;

					MPI_Recv(aTemp, pow(subOrder, 2), MPI_INT, j, 10,
							MPI_COMM_WORLD, &status);

					for (x = 0; x < subOrder; x++) {
						for (y = 0; y < subOrder; y++) {
							a[x + xOffset][y + yOffset] = aTemp[x][y];
						}
					}
				}
				for (x = 0; x < subOrder; x++) {
					for (y = 0; y < subOrder; y++) {
						a[x][y] = aLocal[x][y];
					}
				}

				printCells(a, i);

			} else {
				// Processes other than 0 send own subarrays
				MPI_Send(aLocal, pow(subOrder, 2), MPI_INT, 0, 10,
						MPI_COMM_WORLD);
			}

			/*
			 sprintf(filename, "state_%d", i);	// generate filename

			 // In rank 0, generate file of ORDER * ORDER blank spaces
			 if (rank == 0) {
			 f = fopen(filename, "w");
			 for (j = 0; j < ORDER; j++) {
			 for (k = 0; k < ORDER; k++) {
			 fputs(" ", f);
			 }
			 fputs("\n", f);
			 }
			 fclose(f);
			 }

			 // Open file to write subarray
			 f = fopen(filename, "r+");
			 xOffset = coords[1] * subOrder;
			 yOffset = coords[0] * subOrder;
			 // Loop through cells in subarray
			 for (x = 0; x < subOrder; x++) {
			 for (y = 0; y < subOrder; y++) {
			 // If cell is alive, write "O" in corresponding location in file
			 if (aLocal[x][y] == 1) {
			 fseek(f, (ORDER * (xOffset + x)) + (yOffset + y) + 1,
			 SEEK_SET);
			 fputs("O", f);
			 } else {
			 fputs(" ", f);
			 }
			 }
			 fclose(f);
			 */
		}
	}

// Stop keeping time
	time2 = MPI_Wtime();

	if (rank == 0) {
		// Display time
		printf(
				"Simulation (matrix size %d*%d, %d iterations) took %f seconds to complete.\n",
				ORDER, ORDER, DURATION, time2 - time1);
	}

// Shut down MPI
	MPI_Finalize();

	return 0;
}
