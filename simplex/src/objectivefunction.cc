/*
This file is part of C++lex, a project by Tommaso Urli.

C++lex is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

C++lex is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with C++lex.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "objectivefunction.h"
#include "simplex.h"

namespace optimization {

    /*
        ObjectiveFunction
        =================
        Class that represents an optimization objective function.
        
    */
    
    ObjectiveFunction::ObjectiveFunction() {}
    
    ObjectiveFunction::ObjectiveFunction ( ObjectiveFunctionType type, RowVector const & costs ) :
        type(type),
        costs(costs) {
        
    }
    
    ObjectiveFunction& ObjectiveFunction::operator=( ObjectiveFunction const & objective_function ) {
    
        type = objective_function.type;
        costs = objective_function.costs;
    
        return *this;
    }
    
    void ObjectiveFunction::log() const {
        
        if (type == OFT_MINIMIZE)
            std::cout << "min ( ";
        else
            std::cout << "max ( ";
        
        for (int i = 0; i < costs.cols(); ++i)
            std::cout << costs(i) << "  ";
        std::cout << ") * x " << std::endl;
        
    }

    // todo: change return type to auto
    RowVector ObjectiveFunction::get_value( RowVector const & x) const {
        return costs * x;
    }
    
    void ObjectiveFunction::add_column(long double value) {
        costs.conservativeResize(1, costs.size()+1);
        costs(costs.size()-1) = value;
    }

}

