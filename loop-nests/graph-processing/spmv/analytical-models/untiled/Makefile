# See LICENSE.txt for license details.

ifneq ($(DEBUG),1)
CXX_FLAGS += -std=c++11 -O3 -Wall
else
CXX_FLAGS += -std=c++11 -O0 -g -Wall
endif
PAR_FLAG = -fopenmp

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = untiled-offchip 
SUITE = $(KERNELS) 

.PHONY: all
all: $(SUITE)

% : %.cc *.h
	$(CXX) $(CXX_FLAGS) $< -o $@



.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
