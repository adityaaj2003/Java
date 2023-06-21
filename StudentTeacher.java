import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
class Student{
    String Name,USN;
    int IA1,IA2,IA3,CTA,CIE;
}
class Clerk
{
    public void Entry(Student who,File f)
    {
        NameEntry(who,f);
        
    }
    private void NameEntry(Student s,File f)
    {
        Scanner Scan = new Scanner(f);
        System.out.println("Enter name of the student:");
        s.Name = Scan.next();
        System.out.println("Enter USN of the student:");
        s.USN = Scan.next();
        Scan.close();
    }
}
class Teacher
{
    String Name,Designation;
    public void MarksEntry(Student S,File f) 
    {
        Scanner Scan = new Scanner(f);
        System.out.println("Enter IA1 marks of the student:");
        S.IA1 = Scan.nextInt();
        System.out.println("Enter IA2 marks of the student:");
        S.IA2 = Scan.nextInt();
        System.out.println("Enter IA3 marks of the student:");
        S.IA3 = Scan.nextInt();
        System.out.println("Enter CTA marks of the student:");
        S.CTA = Scan.nextInt();
        Scan.close();
    }
    void ComputeCIE(Student S,File f)
    {
        int small=S.IA1;
        if(S.IA2<small)
        small=S.IA2;
        if(S.IA3<small)
        small=S.IA3;
        S.CIE=S.IA1+S.IA2+S.IA3+S.CTA-small;
        System.out.println("CIE marks of the student is: "+S.CIE);
    }
    // void ToppperList()
}
public class StudentTeacher {
    public static void main(String args[]) throws FileNotFoundException
    {
       int i;
       File f = new File("CIE.txt");
       Scanner Scan = new Scanner(f);
       Student s1[] = new Student[5];
       Clerk C = new Clerk();
       for(i=0;i<5;i++)
       {
        s1[i] = new Student();
        C.Entry(s1[i],f);
       }
       Teacher T = new Teacher();
       T.Name="GMS";
       T.Designation="AP";      
       for(i=0;i<5;i++)
       {
        s1[i] = new Student();
        T.MarksEntry(s1[i],f);
        T.ComputeCIE(s1[i],f);
       }
       Scan.close();
    }

}
