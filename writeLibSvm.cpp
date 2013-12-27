#include "util.h"

int main()
{
	// use this for partial training writePartialLicensesLIBSVMFormat("partial_train.txt");
	writePartialUnlabeledLicensesLIBSVMFormatAndNames("partial_test.txt", "partial_test.names.txt");

	return 0;
}
