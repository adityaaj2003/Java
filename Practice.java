class Student
{
    protected int roll_no;
    public void Enter(Student s)
    {
        s.roll_no=1010;
    }
    void display(Student s)
    {
        System.out.println("Roll number is "+s.roll_no);
    }
}
public class Practice extends Student{
    public static void main(String args[]){
        Student s = new Student();
        s.roll_no=1111;
        System.out.println("Roll number is "+s.roll_no);
        Student s1 = new Student();
        if(s1.roll_no==0)
        s1.Enter(s1);
        s1.display(s1);
    }
}