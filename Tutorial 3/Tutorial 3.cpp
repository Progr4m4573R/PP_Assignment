/*
Assignment Summary 

My Code utilises multiple buffers on the host device to store summary statistics and prevent overwritting of data.
Sum and mean are calculated using the reduce_2 kernel and the sort bitonic kernel from the lecture is used to sort the array which is then processed by my buffer_C for a min and max kernel which simply gets the first and last elements of the sorted buffer from the reduce_2 kernel.
To calculate variance i created a kernel based on the reduce_2 and called it parallel_variance, this was used to sum up the values and calculate the mean by dividing the sum by the size of the data in buffer_A. 
parallel_variance then called another new kernel called myvariance. The myvariance kernel replaces the data in a new buffer with the data in buffer_A subtract the mean and squared. And then the for loop from the reduce_2 kernel is used to sum up the squareed differences.
The squared differences are outputted into the 0 index of the new buffer and can then be used to calculate standard deviation.
The quartiles are calculated by getting the size of the buffer and adding 1 then multiplying by 1/4, for the lower quartile and multiplying by 3/4 for the upper quartile. The Median is the value in the index of the size of the buffer + 1 then divided by 2

Bitesize.Analysing Data. Available at:
https://www.bbc.co.uk/bitesize/guides/zwhgk2p/revision/4#:~:text=To%20find%20the%20median%20value,1%20and%20divide%20by%204.[Accessed 06/05/2021]
StackOverflow Reading a txt file into Vector. Available at:
https://www.cplusplus.com/forum/beginner/264090/
CalculatorSoup. Variance Calculator Available at:
https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php [Accessed 06/05/2021]
CMP3752M Parallel Programming. Dr Mubeen Ghafoor.Sorting.Available at:
https://blackboard.lincoln.ac.uk/webapps/blackboard/execute/content/file?cmd=view&content_id=_4077053_1&course_id=_153444_1 [Accessed 06/05/2021]
StackOverflow How to convert Char into Float. Available at:
https://stackoverflow.com/questions/18494218/how-to-convert-char-into-float [Accessed 06/05/2021]
Opencl in action. Available at:
https://livebook.manning.com/book/opencl-in-action/chapter-4/87 [Accessed 10/05/2021]
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include "Utils.h"

#include <stdio.h>/*printf*/
#include <math.h> /*sqrt*/
		//-----------------------------------------------------------Section 1-----------------------------------------------------------------------------------------
		//1.1 Create memory locations for data to be read into from textfile

vector<float> Air_Temperature;// length of the textfile

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}
		//1.2 Read in the columns from the textfile into the right memory location type
int main(int argc, char **argv) {
	using TYPE = string;
	string filename = "C:/Users/Ghost/OneDrive - University of Lincoln/Year 3 Second Half/Parallel Programming/temp_lincolnshire_datasets/temp_lincolnshire_short.txt";

	vector< vector<TYPE> > data;// vector of vectors for each column to be separate

	ifstream in(filename);// take in the filename as the in stream
	for (string line; getline(in, line); )// for each line of the textfile
	{
		stringstream ss(line);// parse the line into the stream
		vector<TYPE> row;// create a row vector which will separate the things on each like as strings so this will be a string containing strings

		for (TYPE d; ss >> d; ) row.push_back(d);// we bitshift the data with the sstream to a string type d and add that to the row string vector 
		data.push_back(row);// and then we put this row string vector in the data vector making a string of strings 
	}

	//cout << "Accessing and printing the data:\n";
	for (auto& row : data)// for the string rows in the string data
	{
		float f = stof(row[5]);// convert the strings in the 5th index or 5th column to floats
		Air_Temperature.push_back(f);// append them to my Temperature vector
		//cout << f << endl;// print to show values are correct
	}
		
	//handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {

		//host operations
		//Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		
		//1.3 Enable the profiling command so we can request information about data transfer speeds 
		
		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//1.4 create events to attach to a queue command responsible for the kernel launch this will enable us to get event launch times
		cl::Event prof_event;
		cl::Event A_event;
		cl::Event B_event;
		cl::Event C_event;
		cl::Event D_event;

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		//1.5 change the int to a float when testing  float datatypes
		typedef float mytype; 

		//1.6 Different size arrays and data values for testing

		//memory allocation
		//host - input
		std::vector<mytype> Temperature = {6,5,4,2,9,1,3,8,7,6,4,7,2,7,8,9,6,23,0,54,76,34,98,13,42,32,64,76,98,81,21,10};//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
		//std::vector<mytype> Temperature(10000,2);

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		
		//1.7 Setup all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "reduction_complete");
		cl::Kernel kernel_sort = cl::Kernel(program, "sort_bitonic");
		cl::Kernel kernel_minmax = cl::Kernel(program, "min_max");
		cl::Kernel kernel_variance = cl::Kernel(program, "parallel_variance");

		//1.8 manually set the recommended value for the work group sizes by reading a kernel property:
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // get device
		cl_device_id d = device();
		cl_uint float_width;
		cerr << kernel_1.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << ":Is the multiple for determining workgroup sizes that ensures best performance"<< endl; // get info
		cerr << kernel_1.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << ": Is the Maximum work group size to execute the given kernel on the given device" << endl;
		
		//get the preferred vector width which is important for floating point processing
		cerr << clGetDeviceInfo(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(float_width), &float_width, NULL) << ": Is the number of scalars of the given datatype that can fit in a vector " << endl;
	
		//1.9 Set the local_size to the Maximum recommended specific to your system 
		size_t local_size = kernel_1.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);// The number of work items in a work group
		size_t global_size = Temperature.size();
		size_t padding_size = Temperature.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<float> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			Temperature.insert(Temperature.end(), A_ext.begin(), A_ext.end());
		}

		//1.10 get the size of the Array we are using
		size_t input_elements = Temperature.size();//number of input elements
		size_t input_size = Temperature.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

        //-----------------------------------------------------------Section 2-----------------------------------------------------------------------------------------

		//2.1 host - output

		//used for sum and mean
		std::vector<mytype> B(input_elements);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes
		//Used for min and max
		std::vector<mytype> C(input_elements);
		//Used for variance
		std::vector<mytype> D(input_elements);
		//temporary memory location
		std::vector<mytype>partial_sums(local_size);
		//2.2 device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_local(context, CL_MEM_READ_WRITE, local_size);
		//2.3 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &Temperature[0]);
		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, input_size, &Temperature[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_local, 0, 0, local_size);//zero buffer_local buffer on device memory

		//2.4 set arguments for kernels and execute
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_local);
		kernel_1.setArg(2, buffer_B);

		kernel_sort.setArg(0, buffer_A);

		kernel_minmax.setArg(0, buffer_A);
		kernel_minmax.setArg(1, buffer_C);

		kernel_variance.setArg(0, buffer_A);
		kernel_variance.setArg(1, buffer_D);
		
		//kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

        //-------------------------------------------------------Section 3-------------------------------------------------------------------------------------

		//3.1 call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_minmax, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_variance, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//3.2 Attach events to the kernels so we can track how long the process takes
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueNDRangeKernel(kernel_minmax, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueNDRangeKernel(kernel_variance, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

		//print array before copying the results from the sorting function kernel
		std::cout << "Unsorted Temperature = " << Temperature << std::endl;
		       
		//3.3 Copy the results of kernel functions from device to host \\ must come before we attach the events in 3.4 or we get an error
		queue.enqueueReadBuffer(buffer_local, CL_TRUE, 0, local_size, &partial_sums[0]);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &D[0]);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &Temperature[0]);

		//3.4 large datasets measure the upload time from host to device for input vector A and download time for output vector B
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &Temperature[0], NULL, &A_event);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, input_size, &B[0], NULL, &B_event);
		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, input_size, &Temperature[0], NULL, &C_event);
		queue.enqueueWriteBuffer(buffer_D, CL_TRUE, 0, input_size, &D[0], NULL, &D_event);
		queue.enqueueWriteBuffer(buffer_local, CL_TRUE, 0, local_size, &partial_sums[0], NULL);
		//--------------------------------------------------Section 4 --------------------------------------------------------------------------------
		//4.1 print out the results
		std::cout << "Sorted Temperature = " << Temperature << std::endl;
		std::cout << "Sum = "<<B[0]<<" Mean = " << B[0]/Temperature.size() << std::endl;
		std::cout << "Min = "<< C[1] <<" Max = "<< C[0]<< std::endl; 
		std::cout << "Median = " << Temperature[floor((Temperature.size()+1)/2)] << endl;
		std::cout << "Upper Quartile = " << Temperature[(Temperature.size()+1)*0.75] << " Lower Quartile = " << Temperature[(Temperature.size()+1)/4] << endl;
		std::cout << "Variance = " << D[0] <<" Standard Deviation = " << sqrt(D[0])<<endl;
		//4.2 Display the kernel execution time and buffer write speeds at the end of the program:
		std::cout << "Kernel execution time [ns]:" <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Buffer A write time [ns]:" <<
			A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Buffer B write time [ns]:" <<
			B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Buffer C write time [ns]:" <<
			C_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			C_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		
		std::cout << "Buffer D write time [ns]:" <<
			D_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			D_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		//4.3 get full information about each profiling event including enqueueing and preparation time
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US)
			<< std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}