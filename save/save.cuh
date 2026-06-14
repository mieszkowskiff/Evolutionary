#ifndef SAVE_HPP
#define SAVE_HPP

#include <iostream>
#include "creatures/creatures.cuh"
#include "creatures/helpers.cuh"



class SaveManager {
    public:
    SaveManager() = default;
    ~SaveManager() = default;

    void Save(const Creatures* creatures, int start_creature_index);

    void SaveMap(const Map* map);
};


#endif // SAVE_HPP