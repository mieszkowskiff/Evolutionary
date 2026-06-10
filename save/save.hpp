#ifndef SAVE_HPP
#define SAVE_HPP

#include <iostream>
#include "creatures/creatures.cuh"

class SaveManager {
    public:
    SaveManager() = default;
    ~SaveManager() = default;

    void Save(const Creatures* creatures) {
        std::cout << "Saving " << creatures->count << " creatures..." << std::endl;
    }
};


#endif // SAVE_HPP