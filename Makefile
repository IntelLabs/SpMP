LOADIMBA=0
LOADIMBA=${loadimba}

LIBRARY		= libspmp.a

ifdef CUSTOM_CXX
  CXX   = $(CUSTOM_CXX)
  # Assume gcc
  CXXFLAGS		= -fopenmp -std=c++11 -Wno-deprecated -Wall
else
  CXX		= icpc
  CXXFLAGS		= -qopenmp -std=c++11 -Wno-deprecated -Wall
endif

ifeq (yes, $(DBG))
  CXXFLAGS += -O0 -g
else
  CXXFLAGS += -O3 -DNDEBUG
endif

ifeq (yes, $(XEON_PHI))
  CXXFLAGS	+= -mmic
else
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  else
    ifndef CUSTOM_CXX
      CXXFLAGS += -xHost
    endif
  endif
endif

ifeq (yes, $(MKL))
  ifndef CUSTOM_CXX
    CXXFLAGS += -mkl
  endif
  CXXFLAGS += -DMKL
endif

ifeq (${LOADIMBA}, 1)
  yesnolist += LOADIMBA
endif

DEFS += $(strip $(foreach var, $(yesnolist), $(if $(filter 1, $($(var))), -D$(var))))
DEFS += $(strip $(foreach var, $(deflist), $(if $($(var)), -D$(var)=$($(var)))))

CXXFLAGS 	+= ${DEFS}
SRCS = $(wildcard *.cpp) $(wildcard reordering/*.cpp)
SYNK_SRCS = $(wildcard synk/*.cpp)
OBJS = $(SRCS:.cpp=.o) $(addsuffix .o, $(addprefix synk/, $(basename $(notdir $(SYNK_SRCS)))))

$(LIBRARY): $(OBJS)
	rm -f $(LIBRARY)
	ar qvs $(LIBRARY) $(OBJS) 

all: clean
	$(MAKE) $(LIBRARY)

test: test/gs_test test/reordering_test test/trsv_test

test/%: test/%.o $(LIBRARY)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

clean:
	rm -f $(LIBRARY) $(OBJS) test/*.o
