CXX?=g++
CFLAGS=-O3 -Wall -Wextra -Wno-ignored-attributes -Wunused-parameter -Wpointer-arith -DLARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -funroll-loops
COMPILATION_RESOURCES=-ftemplate-depth=1000000 -fconstexpr-depth=1000000 -march=native -fconstexpr-ops-limit=999999999 -fconstexpr-loop-limit=9999999 -mcmodel=medium
CPPFLAGS=-I ../src
LD=${CXX}
LDLIBS=-lm -lz -lpthread -lstdc++
CXXFLAGS=-std=c++20
OPENMP=-fopenmp

ifeq (,$(findstring g++,$(CXX)))
  CFLAGS+=-Werror
	COMPILATION_RESOURCES+=-fconstexpr-ops-limit=999999999
endif

ifdef prof
	CFLAGS+=-pg -g -fprofile-arcs -ftest-coverage
	LDLIBS+=-pg -g -fprofile-arcs -ftest-coverage
endif

ifdef debug
	CFLAGS+=-g
	LDLIBS+=-g
endif

ifdef time
	CFLAGS+=-D_TIME
endif

#ifeq (,$(findstring clang++,$(CXX)))
#	COMPILATION_RESOURCES+=-fconstexpr-steps=99999999
#endif

GTTL?=../gttl
GTTLFLAGS=-I ${GTTL}/src

LIBTORCH?=../libtorch
LIBTORCHFLAGS_I=-I ${LIBTORCH}/include -I ${LIBTORCH}/include/torch/csrc/api/include
LIBTORCHFLAGS_L=-L ${LIBTORCH}/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,${LIBTORCH}/lib

FAISS?=/home/stud2018/ata/faiss
FAISSFLAGS_I=-I /home/stud2018/ata/cuda/include -I ${FAISS}/include
FAISSFLAGS_L=-L ${FAISS}/lib64 -L /home/stud2018/ata/cuda/lib64 -L /home/stud2018/ata/openblas/lib/ -lfaiss -lcublas -lcudart -lcudadevrt -lopenblas -lcusolver -lcublasLt

DEBUG=-Wl,-t -Wl,--verbose

TMPFILE=temp.txt
REFDIR?=../testdata/ref

SQGRAM=sorted_qgram
UQGRAM=unsorted_qgram
SPACED_SEEDS=spaced_seeds
SMATRIX=score_matrix
ITER=blast_iterate

all:${SQGRAM}.x ${ITER}.x ${SMATRIX}.x ${UQGRAM}.x

test: test_unsorted_qgram test_sorted_qgram test_spaced_seeds test_iterate 

test_sorted_qgram:${SQGRAM}.x
	@./${SQGRAM}.x -q 3 -s >${TMPFILE}
	@diff ${REFDIR}/sorted_qgram_nuc_3.txt ${TMPFILE}
	@./${SQGRAM}.x -p -q 2 -s >${TMPFILE}
	@diff ${REFDIR}/sorted_qgram_prot_2.txt ${TMPFILE}
	@${RM} ${TMPFILE}
	@echo "$@ passed."

test_unsorted_qgram:${UQGRAM}.x
	@./${UQGRAM}.x -q 3 -s >${TMPFILE}
	@diff ${REFDIR}/unsorted_qgram_nuc_3.txt ${TMPFILE}
	@./${UQGRAM}.x -p -q 2 -s >${TMPFILE}
	@diff ${REFDIR}/unsorted_qgram_prot_2.txt ${TMPFILE}
	@${RM} ${TMPFILE}
	@echo "$@ passed."

test_spaced_seeds:${SPACED_SEEDS}.x
	@./check_err.py ./${SPACED_SEEDS}.x
	@./${SPACED_SEEDS}.x -p -d 0 -s ${GTTL}/testdata/sw175.fna >${TMPFILE}
	@diff ${REFDIR}/spaced_seeds_sw175.txt ${TMPFILE}
	@./${SPACED_SEEDS}.x -p -d 7 -s ${GTTL}/testdata/sw175.fna >${TMPFILE}
	@diff ${REFDIR}/spaced_seeds_sw175_7.txt ${TMPFILE}
	@${RM} ${TMPFILE}
	@echo "$@ passed."

test_iterate: ${ITER}.x
	@./check_err.py ./${ITER}.x
	@./${ITER}.x -s ${GTTL}/testdata/sw175.fna >${TMPFILE}
	@diff ${REFDIR}/blast_iterate_sw175_ALL_0.txt ${TMPFILE}
	@./${ITER}.x -d 7 -s ${GTTL}/testdata/sw175.fna >${TMPFILE}
	@diff ${REFDIR}/blast_iterate_sw175_ALL_7.txt ${TMPFILE}
	@echo "$@ passed."

%.o:%.cpp
	$(CXX) -o $@ -c -fdiagnostics-all-candidates $< ${LIBTORCHFLAGS_I} ${FAISSFLAGS_I} ${OPENMP} ${CXXFLAGS} ${COMPILE_CONST} ${COMPILATION_RESOURCES} ${CFLAGS} ${CPPFLAGS} ${GTTLFLAGS} ${DEBUG} -MT $@ -MMD -MP -MF $(@:.o=.d)

%.x:%.o
	$(CXX) ${LDFLAGS} $< -o $@ ${LIBTORCHFLAGS_L} ${FAISSFLAGS_L} ${LDLIBS} ${OPENMP}

.PHONY:clean
clean:
	${RM} -r *.[odx] *.x.dSYM/ __pycache__ temp.txt
