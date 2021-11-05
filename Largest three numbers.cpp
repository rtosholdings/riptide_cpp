#include <iostream>

using namespace std;

int main()
{
    int arr[3],i;
    int maximum = arr[0];
    int third=arr[0],second=arr[0];
    for (i=0; i<=3 ; i++)
    {
        cout<<"The elements in the array are:"<<endl;
        cin>>arr[i];
    }
    for(i = 0; i <= 3; i++)
    {
        if(maximum < arr[i])
        {

            third = second;
            second = maximum;
            maximum = arr[i];

        }
        else if (arr[i] > second)
        {
            third = second;
            second = arr[i];
        }
        else if (arr[i] > third)
        {
            third = arr[i];

        }

    }
    cout<<maximum<<second<<third<<endl;
    return 0;
}
