# Compiler
NVCC = nvcc

# Include directory
INCLUDES = -Iinclude

# Source files
SRCS = kernel_2.cu src/savePPM.cpp

# Output executable
TARGET = kernel

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(INCLUDES) $(SRCS) -o $(TARGET)
	
run: $(TARGET)	
	chmod u+x $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)