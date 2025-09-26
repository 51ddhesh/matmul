#pragma once
#include <chrono>
#include <random>

namespace random_ {
    inline std::mt19937& getEngine() {
        static std::mt19937 engine(
            static_cast<unsigned int> (std::chrono::steady_clock::now().time_since_epoch().count())
        );
        return engine;
    }
    inline int getInt(int min, int max) {
        static std::uniform_int_distribution<int> dist(min, max);
        return dist(getEngine());
    }
} // namespace random_
