#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;

int main()
{
	int a, b, y, m;

	cout << "Enter no. of training samples to be generated: ";
	cin >> m;

	ofstream file;
	file.open("training-data-xor.csv");

	file << "A,B,Y"; // Header
	for (int i = 0; i < m; ++i) {
		a = rand() % 2;
		b = rand() % 2;
		y = a ^ b;
		file << "\n" << a << "," << b  << "," << y;
	}

	file.close();

	cout << "File [training-data-xor.csv] generated." << endl;
}