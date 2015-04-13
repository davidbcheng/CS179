// void test1(){
//     int *a = 3;
//     *a = *a + 2;
//     printf("%d",*a);
// }

// void test1(){
//     int *a = (int *) malloc(sizeof(int));
//     *a = 3;
//     *a = *a + 2;
//     printf("%d",*a);
// }

void test1() {
	int temp = 3;
	int *a = &temp;
	*a = *a + 2;
	printf("%d",*a);
}

int main() {
	test1();
	return 0;
}