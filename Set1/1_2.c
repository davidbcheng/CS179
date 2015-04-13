void test2(){
    int* a, *b;
    a = (int*) malloc(sizeof(int));
    b = (int*) malloc(sizeof(int));

    if (!(a && b)){
        printf("Out of memory");
        exit(-1);
    }
    *a = 2;
    *b = 3;

    printf("a: %d, b:%d\n", *a, *b);
}

void test3(){
    int i, *a = (int*) malloc(1000 * sizeof(int));

    if (!a){
        printf("Out of memory");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i+a)=i;

    for (i = 0; i < 1000; i++)
        printf("%d\n", *(i+a));
}

void test4(){
    int **a = (int**) malloc(3*sizeof(int*));
    int i;

    if (!a) {
        printf("Out of memory");
        exit(-1);
    }

    for(i = 0; i < 3; i++) {
        a[i] = (int *) malloc(100 * sizeof(int*));
    }

    a[1][1] = 5;
}

void test5(){
    int *a = (int*) malloc(sizeof(int));
    scanf("%d", a);
    if (!(*a))
        printf("Value is 0\n");
}

int main(int argc, char const *argv[])
{
	test5();
	return 0;
}