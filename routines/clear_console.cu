#include <iostream>


void clearConsole() {
    std::cout << "\033[2J\033[1;1H";
}


int main() {
    clearConsole();
    
    return 0;
}