import java.sql.Array;
import java.util.LinkedList;
import java.util.Scanner;

import javax.swing.text.html.HTMLDocument.Iterator;

public class TW2 {
    public static void main(String args[]){
        LinkedList L = new LinkedList();
        Scanner Scan = new Scanner(System.in);
        int ele,choice,ele1,ele2,n;
        while(true)
        {
            System.out.println("Enter choice 1.Insert 2.Delete 3.Display 4.Sort 5.Exit");
            choice=Scan.nextInt();
            switch(choice)
            {
                case 1:System.out.println("Enter elemnt to insert");
                        ele=Scan.nextInt();
                        L.addLast(ele);
                        break;
                case 2:System.out.println("Element deleted is "+L.poll());
                       break;
                case 3:System.out.println("Elements of the list are:"+L);
                       break;
                case 4:n=L.size();
                for(int i=0;i<n-1;i++)
                 {
                    for(int j=0;j<n-1-i;j++)
                    {
                        ele1=(int)L.get(j);
                         ele2=(int)L.get(j+1);  
                         if(ele1>ele2)                       
                         {
                            L.set(j,ele2);
                            L.set(j+1,ele1);
                         }
                    }
                 }
                 System.out.println("After sorting:"+L);
                case 5:return;
            }
        }
        // Scan.close();
    }
}
