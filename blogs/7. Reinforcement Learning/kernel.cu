#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iomanip>
#include <thread>
#include <chrono>

struct Position { int x, y; };

struct EnvDevice {
    char map[16];
    __device__ __host__ int pos_to_index(Position p) const { return p.y * 4 + p.x; }

    __device__ __host__ void step(Position& pos, int action, float& reward, bool& done) const {
        int nx = pos.x, ny = pos.y;
        if (action == 0) nx = max(0, pos.x - 1);      // Left
        else if (action == 1) ny = min(3, pos.y + 1); // Down
        else if (action == 2) nx = min(3, pos.x + 1); // Right
        else if (action == 3) ny = max(0, pos.y - 1); // Up

        pos.x = nx; pos.y = ny;
        int idx = pos_to_index(pos);
        if (map[idx] == 'H') { reward = 0.0f; done = true; }      // Hole
        else if (map[idx] == 'G') { reward = 1.0f; done = true; } // Goal
        else { reward = 0.0f; done = false; }                     // Frozen path
    }
};

// Function to visualize the agent's movement on the console
void render_simulation(const EnvDevice& env, const float* q_table, const std::string& title) {
    Position pos = { 0, 0 };
    bool done = false;
    int steps = 0;

    while (!done && steps < 15) {
#ifdef _WIN32
        system("cls");
#else
        system("clear");
#endif
        std::cout << "  " << title << std::endl;
        std::cout << "========================================" << std::endl;

        for (int r = 0; r < 4; r++) {
            std::cout << "      ";
            for (int c = 0; c < 4; c++) {
                if (pos.y == r && pos.x == c) std::cout << "A "; // A = Agent
                else std::cout << env.map[r * 4 + c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "========================================" << std::endl;

        int state_idx = env.pos_to_index(pos);
        int action;

        // If q_table is null, perform random walk (Pre-training demo)
        if (q_table == nullptr) {
            action = rand() % 4;
        }
        else {
            // Select the best action based on learned Q-values (Exploitation)
            action = 0;
            float max_q = q_table[state_idx * 4];
            for (int i = 1; i < 4; i++) {
                if (q_table[state_idx * 4 + i] > max_q) {
                    max_q = q_table[state_idx * 4 + i];
                    action = i;
                }
            }
        }

        float r_unused; bool d_unused;
        env.step(pos, action, r_unused, d_unused);
        if (env.map[env.pos_to_index(pos)] == 'H' || env.map[env.pos_to_index(pos)] == 'G') done = true;

        steps++;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    if (env.map[env.pos_to_index(pos)] == 'G') std::cout << "\n   >>> STATUS: SUCCESS! <<<\n" << std::endl;
    else std::cout << "\n   >>> STATUS: FAILED! <<<\n" << std::endl;
}

// CUDA Kernel for parallel Q-Learning
__global__ void q_learning_kernel(float* q_table, int* success_history, EnvDevice env, unsigned long seed, float alpha, float gamma, int episodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);

    for (int ep = 0; ep < episodes; ++ep) {
        Position pos = { 0, 0 };
        bool done = false;
        int steps = 0;
        while (!done && steps < 40) {
            int state_idx = env.pos_to_index(pos);

            // Epsilon-greedy: 10% chance for a random move
            int action = (curand_uniform(&state) < 0.1f) ? (curand(&state) % 4) : 0;
            if (action == 0) {
                float max_val = q_table[state_idx * 4];
                for (int i = 1; i < 4; i++) {
                    if (q_table[state_idx * 4 + i] > max_val) {
                        max_val = q_table[state_idx * 4 + i];
                        action = i;
                    }
                }
            }

            float reward;
            env.step(pos, action, reward, done);

            // Record success for statistics
            if (reward > 0.5f) atomicAdd(&success_history[ep], 1);

            // Bellman Equation: Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            int next_idx = env.pos_to_index(pos);
            float max_next = q_table[next_idx * 4];
            for (int i = 1; i < 4; i++) max_next = fmaxf(max_next, q_table[next_idx * 4 + i]);

            float old_q = q_table[state_idx * 4 + action];
            float new_q = old_q + alpha * (reward + gamma * max_next - old_q);

            // Thread-safe update of the global Q-table
            atomicExch(&q_table[state_idx * 4 + action], new_q);
            steps++;
        }
    }
}

int main() {
    EnvDevice h_env;
    // Map layout: S=Start, F=Frozen, H=Hole, G=Goal
    char m[16] = { 'S','F','F','F','F','H','F','H','F','F','F','H','H','F','F','G' };
    memcpy(h_env.map, m, 16);

    float* d_q_table;
    int* d_success_history;
    const int total_episodes = 10000;

    cudaMalloc(&d_q_table, 64 * sizeof(float));
    cudaMalloc(&d_success_history, total_episodes * sizeof(int));

    float alpha;
    const float gamma = 0.95f;

    // Show initial random behavior
    render_simulation(h_env, nullptr, "========== REINFORCEMENT LEARNING CUDA ==========\nDeveloped by: Natan Jarzynski (TheNatiMix) & Marek Kowalik (Kowalikov)\n\nINITIAL RANDOM WALK (NO KNOWLEDGE)");
    std::cout << "\nInitial demo finished. Now let's start the training lab.\n";

    while (true) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Enter Alpha (learning rate 0.01-0.99, -1 to exit): ";
        if (!(std::cin >> alpha) || alpha < 0) break;

        // Reset Q-Table and history for a clean state
        cudaMemset(d_q_table, 0, 64 * sizeof(float));
        cudaMemset(d_success_history, 0, total_episodes * sizeof(int));

        std::cout << "GPU is crunching data..." << std::endl;
        const int threadsPerBlock = 256;
        const int blocksPerGrid = 4;
        const int totalAgents = threadsPerBlock * blocksPerGrid;
        q_learning_kernel <<<blocksPerGrid, threadsPerBlock>>> (d_q_table, d_success_history, h_env, (unsigned long)time(NULL), alpha, gamma, total_episodes);
        cudaDeviceSynchronize();

        // Retrieve results from GPU
        std::vector<int> h_success_history(total_episodes);
        cudaMemcpy(h_success_history.data(), d_success_history, total_episodes * sizeof(int), cudaMemcpyDeviceToHost);

        float h_q_table[64];
        cudaMemcpy(h_q_table, d_q_table, 64 * sizeof(float), cudaMemcpyDeviceToHost);

        // Simulate the "intelligent" agent
        render_simulation(h_env, h_q_table, "SMART WALK (LEARNED STRATEGY)");

        // Display statistics
        std::cout << "--- TRAINING STATISTICS FOR ALPHA: " << alpha << " ---" << std::endl;
        for (int i = 0; i < total_episodes; i += 1000) {
            float sum = 0;
            for (int j = i; j < i + 1000; j++) sum += h_success_history[j];
            // Calculation: sum / (Threads * Episodes per Block) * 100
            float rate = (sum / (float)(totalAgents * 1000.0f) * 100.0f);
            std::cout << "Block " << (i / 1000 + 1) << " (Episodes " << i << "-" << i + 1000 << "): " << std::fixed << std::setprecision(2) << rate << "% Success Rate" << std::endl;
        }

        std::cout << "\nPress ENTER to try with another alpha..." << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
    }

    cudaFree(d_q_table);
    cudaFree(d_success_history);
    return 0;
}