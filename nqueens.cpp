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
  std::vector<std::vector<bool>> domains;

  Node(size_t N, Data data): depth(0), board(N), domains(){
    for (int i = 0; i < N; i++) {
      board[i] = i;
      domains.resize(N, std::vector<bool>(data.get_max_u(), true));
      for(int j = 0; j < data.get_max_u(); j++)
        domains[i][j] = true;
    }
  }

  Node(const Node&) = default; //allow to copy an existing node
  Node(Node&&) = default;      //allow to move a node
  Node() = default; 
};

// check if placing a queen is safe (i.e., check if all the queens already placed share
// a same diagonal)
//Modify nqueens.cpp to check an arbitrary list of inequalities. 
//I slightly modify this function to take into account other constraints
//          //current configuration         //row anc column where we have to evaluate


/*
__global__ void update_domains_cuda(bool *domains, int *parent_depth, int *starting_depth, int *j, Data *data ){

  return;
}
*/
//
//now use cuda to parallelize this
void update_domains(std::vector<std::vector<bool>>& domains, int parent_depth, int starting_depth, int j, Data data){
#pragma omp parallel for num_threads(4)  
  for(int i = starting_depth; i < data.get_n(); i++){
    if(data.get_C_at(i, parent_depth) == 1)
      domains[i][j] = false;
  }
  return;
}

bool isSafe(/*const std::vector<int>& board, */const int row, const int col, Data data)
{
  if(data.get_C_at(row, col) == 1)
    return false;

  return true;
}

// evaluate a given node (i.e., check its board configuration) and branch it if it is valid
// (i.e., generate its child nodes.)
void evaluate_and_branch(const Node& parent, std::stack<Node>& pool, size_t& tree_loc, size_t& num_sol, Data data)
{
  int depth = parent.depth;
  int N = parent.board.size();

  // if the given node is a leaf, then update counter and do nothing
  if (depth == N) {
    num_sol++;
  }
  // if the given node is not a leaf, then update counter and evaluate/branch it
  /*
  else{
    for (int j = depth; j < N; j++) {
      if (isSafe(parent.board, depth, parent.board[j], data)) {
        Node child(parent);
        std::swap(child.board[depth], child.board[j]);
        child.depth++;
        pool.push(std::move(child));
        tree_loc++;
      }
    }
  }
  */

  else{
    int upper_bound = data.get_u_at(depth);
    for(int j = 0; j < upper_bound; j++){
      if(parent.domains[depth][j] == true){
        //call update domains
        Node child(parent);
        child.depth++;
        tree_loc++;
        update_domains(child.domains, parent.depth, child.depth, j, data);
        //before call update_domains_cuda is necessary to transfer the necessary struct to the gpu

        //child domains
        /*
        bool *child_domains_gpu; cudaMalloc(&child_domains_gpu, data.get_n() * data.get_max_u() * sizeof(bool)); cudaDeviceSynchronize();
        cudaMemcpy(child_domains_gpu, &child.domains, data.get_n() * data.get_max_u() * sizeof(bool), cudaMemcpyHostToDevice); cudaDeviceSynchronize();

        //parent.depth
        int *parent_depth_gpu; cudaMalloc(&parent_depth_gpu, sizeof(int)); cudaDeviceSynchronize();
        cudaMemcpy(parent_depth_gpu, &(parent.depth), sizeof(int), cudaMemcpyHostToDevice);cudaDeviceSynchronize();

        //child.depth
        int *child_depth_gpu; cudaMalloc(&child_depth_gpu, sizeof(int)); cudaDeviceSynchronize();
        cudaMemcpy(child_depth_gpu, &(child.depth), sizeof(int), cudaMemcpyHostToDevice); cudaDeviceSynchronize();

        //j (column)
        int *j_gpu; cudaMalloc(&j_gpu, sizeof(int)); cudaDeviceSynchronize();
        cudaMemcpy(j_gpu, &j, sizeof(int), cudaMemcpyHostToDevice); cudaDeviceSynchronize();

        //data
        Data *data_gpu; cudaMalloc(&data_gpu, sizeof(Data)); cudaDeviceSynchronize();
        cudaMemcpy(data_gpu, &data, sizeof(Data), cudaMemcpyHostToDevice); cudaDeviceSynchronize();

        //now we can call update_domains_cuda
        //update_domains_cuda<<< 1, 2 >>>(child_domains_gpu, parent_depth_gpu, child_depth_gpu, j_gpu, data_gpu); 
        
        //and then copy back to CPU
        //child domains //we just need to copy back what was actually modified
        cudaMemcpy(&child.domains, child_domains_gpu, data.get_n() * data.get_max_u() * sizeof(bool), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
        */

        //finally we push the child into the stack
        pool.push(std::move(child));
      }
      

    }
  }
  /*
  else {
    for (auto it = parent.possible_places.begin(); it != parent.possible_places.end(); ) {
      if (isSafe(depth, *it, data)) {
          parent.possible_places.erase(it++);
          parent.possible_places.erase(it++);
          Node child(parent);
          child.depth++;
          pool.push(std::move(child));
          tree_loc++;
      }
      else {
        ++it;
      }
  }
}
*/
  }
  
 /*
 //my version with backtracking
  else{
    for(int j = 0; j < parent.possible_places[depth].size(); j++){
      if (parent.possible_places[depth][j] == -1) {
        continue; // Skip invalid positions
      }
      Node child(parent); child.depth++; tree_loc++; pool.push(std::move(child));
      if(depth + 1 != N){
        for(int k = depth + 1; k < N; k++){
          if(data.get_C_at(k, depth) == 1){
            child.possible_places[k][j] = -1;

          }
        }
      }
  }
}
*/


int main(int argc, char** argv) {
    Data data;
    if (data.read_input("pco_3.txt")){
        data.print_n();
        data.print_u();
        data.print_C();
    }

  //test print
  //inline int get_u_at(size_t i){return u[i];}
  std::cout << "u[0]:  " <<data.get_u_at(0) << std::endl;
  std::cout << "MAX:  " <<data.get_max_u() << std::endl;

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
    evaluate_and_branch(currentNode, pool, exploredTree, exploredSol, data);


  }

  //get the finish time
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // outputs
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
  std::cout << "Total solutions: " << exploredSol << std::endl;
  std::cout << "Size of the explored tree: " << exploredTree << std::endl;

  return 0;
}