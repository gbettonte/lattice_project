/*
 * Author: Guillaume HELBECQUE (Université du Luxembourg)
 * Date: 10/10/2024
 *
 * Description:
 * This program solves the N-Queens problem using a sequential Depth-First tree-Search
 * (DFS) algorithm. It serves as a basis for task-parallel implementations.
 */

#include <omp.h>

#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include <chrono>
#include <stack>
#include <set>
#include <algorithm>
#include "parser.hpp"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda/atomic>
using namespace std;




// N-Queens node
struct Node {
  int depth; // depth in the tree
  std::vector<int> board; // board configuration (permutation)
  //Example of permutation with 4 queens: [2, 0, 3, 1] queen of the first row on the second column, and so on
  //std::set<int> possible_places;
  std::vector<bool> domains;

  Node(size_t N, Data data): depth(0), board(N), domains(){
    for (int i = 0; i < N; i++) {
      board[i] = i;
      //domains.resize(N * std::vector<bool>(data.get_max_u(), true));
      int max = data.get_max_u();
      domains.resize(N * max);
      for(int j = 0; j < max; j++)
        domains[i*max+j] = true;
    }
  }

  Node(const Node&) = default; //allow to copy an existing node
  Node(Node&&) = default;      //allow to move a node
  Node() = default; 
};

//this function updates the domains using OpenMP
//each thread is in charge of one iteration of the loop
void update_domains(std::vector<bool>& domains, int parent_depth, int starting_depth, int j, int n, int max_u, int* u, int* array_C){
#pragma omp parallel for num_threads(4)  
  for(int i = starting_depth; i < n; i++){
    if( array_C[i*max_u+j] == 1)
      domains[i*max_u+j] = false;
  }
  return;
}


void evaluate_and_branch(const Node& parent, std::stack<Node>& pool, size_t& tree_loc, size_t& num_sol, int n, int max_u, int* u,  int* array_C)
{
  int depth = parent.depth;
  int N = parent.board.size();

  // if the given node is a leaf, then update counter and do nothing
  if (depth == N) {
    num_sol++;
  }
  // if the given node is not a leaf, then update counter and evaluate/branch if possible
  else{
    int upper_bound = u[depth];
    for(int j = 0; j < upper_bound; j++){
      if(parent.domains[depth*max_u + j] == true){ //if we can we create a child
        Node child(parent);
        child.depth++;
        tree_loc++;
        update_domains(child.domains, parent.depth, child.depth, j, n, max_u, u, array_C); //update of the domains of the child
        //finally we push the child into the stack
        pool.push(std::move(child));
      }
    }
  }
}
  


int main(int argc, char** argv) {
    Data data;
    
    if (data.read_input("pco_3.txt")){
        data.print_n();
        data.print_u();
        data.print_C();
    }

  //get useful and constant information
  int* u = data.get_u();
  int n = data.get_n();
  int max_u = data.get_max_u();
  int** C = data.get_C(); //we retrieve the constraints matrix
  int* array_C = (int*)malloc(n* n * sizeof(int)); // we allocate a 1D array
  
  int i, j;
  for(i = 0; i < n; i++)
    std::cout << u[i];
  std::cout << endl;
  std::cout << endl;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      array_C[i * n + j] = C[i][j];
      std ::cout<< array_C[i * n + j] << " ";
    }
    std::cout << endl;
  }

  // helper
  if (argc != 2) {
    std::cout << "usage: " << argv[0] << " <number of queens> " << std::endl;
    exit(1);
  }

  // problem size (number of queens)
  size_t N = std::stoll(argv[1]);
  std::cout << "Solving " << N << "-Queens problem\n" << std::endl;

  // initialization of the root node (the board configuration where no queen is placed)
  Node root(N, data);

  // initialization of the pool of nodes (stack -> DFS exploration order)
  std::stack<Node> pool; //stack of nodes; a stack is LIFO First IN First OUT
  pool.push(std::move(root)); //push the root on the stack

  // statistics to check correctness (number of nodes explored and number of solutions found)
  size_t exploredTree = 0;
  size_t exploredSol = 0;

  // beginning of the Depth-First tree-Search
  auto start = std::chrono::steady_clock::now();

  while (pool.size() != 0) { //i.e continue till all the the path are explored
    // get a node from the pool
    Node currentNode(std::move(pool.top()));
    pool.pop();

    // check the board configuration of the node and branch it if it is valid.
    evaluate_and_branch(currentNode, pool, exploredTree, exploredSol, n, max_u, u, array_C);


  }

  //get the finish time
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // outputs
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
  std::cout << "Total solutions: " << exploredSol << std::endl;
  //std::cout << "Size of the explored tree: " << exploredTree << std::endl;

  return 0;
}
