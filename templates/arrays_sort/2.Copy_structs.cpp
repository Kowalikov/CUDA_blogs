#include <iostream>

using std::cout;
using std::endl;
using std::string;
using std::to_string;

struct Person {
    string name;
    int age;
    float height;

    string toString() const {
        return "{ name: " + name
             + ", age: " + to_string(age)
             + ", height: " + to_string(height)
             + " }";
    }
};


int main() {
    const int N = 10;

    Person p1 = { "Andrzej", 30, 175.5f };
    cout << p1.toString() << endl;

    Person p2 = p1; //shallow copy
    p2.name = "Zbigniew";
    cout << p1.toString() << endl;
    cout << p2.toString() << endl;

    // deep copy via pointer
    Person* p3 = &p1;
    p3->age = 35;
    cout << p1.toString() << endl;  
    cout << p3->toString() << endl;

    // Deep copy and shallow copy via pointer
    Person* p4pointer = &p1;

    p4pointer->age = 40;
    Person p4 = *p4pointer;
    p4pointer->height = 180.0f;

    cout << p1.toString() << endl; 
    cout << p4.toString() << endl;

    return 0;
}