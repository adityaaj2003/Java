//Program to sort linked list 
#include<stdio.h>

struct node
{
    int data;
    struct node *link;
};
typedef struct node *NODE;
NODE create_node()
{
    NODE ptr=(NODE)malloc(sizeof(struct node));
    return ptr;
}
NODE insertfront(NODE first,int ele)
{
    NODE temp=create_node();
    temp->data=ele;
    temp->link=NULL;
    if(first==NULL)
    return temp;
    temp->link=first;
    return temp;
}
NODE deleterear(NODE first)
{
    NODE cur=NULL;
    NODE prev=NULL;
    cur=first;
    if(first==NULL)
    {
      printf("List is empty\n");
      return NULL;
    }
    while(cur->link==NULL)
    {
      prev=cur;
      cur=cur->link;
    }
    prev->link=NULL;
    
    printf("Element deleted is %d\n",cur->data);
    free(cur);
}
void display(NODE first)
{
    NODE cur=first;
    if(first==NULL)
    {
        printf("List is empty\n");
        return;
    }
    while(cur!=NULL)
    {
        printf("%d--",cur->data);
        cur=cur->link;
    }
    printf("\n");
}
main()
{
    NODE first=NULL;
    int choice,ele,key;
    while(1)
    {
      printf("Enter choice\n");
      scanf("%d",&choice);
      switch(choice)
      {
        case 1:printf("Enter element to insert\n");
               scanf("%d",&ele);
               first=insertfront(first,ele);
               break;
        case 2:first=deleterear(first);                                              
                break;
         case 3:printf("Contents are\n");
               display(first);
               break;
       
        default:printf("Invalid\n");
      }
    }
}