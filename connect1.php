<?php
   $connect=mysqli_connect("localhost","root","","dbms");

   if(isset($_POST['submit'])){
    $id=$_POST['id'];
    $name=$_POST['name'];
    $number=$_POST['number'];

    $sql="INSERT INTO labdbms(ID,Name,Phone) values ('$id','$name','$number')";
    $result=mysqli_query($connect,$sql);

    if($result==TRUE){
        echo"Data is inserted";
    }
    else{
        echo" Error Occured";
    }
   }

   if(isset($_POST["select"])){

    $query="SELECT * FROM labdbms";
    $result=mysqli_query($connect,$query);
    if($result==true){
        echo "All  Registered Students List <br>";
    echo "<table cellpadding=10 border='1'>
    <tr>
    <th>ID</th> 
    <th>Name</th>
    <th>Phone Number</th>
    </tr>";
     if(mysqli_num_rows($result) > 0)
    {
    while($row = mysqli_fetch_array($result))
    {	
    echo "<tr>";
    echo "<td style='color:black'>" . $row['ID'] ."</td>";
    echo "<td style='color:black'>" . $row['Name'] . "</td>";
    echo "<td style='color:black'>" . $row['Phone'] . "</td>";
    echo "</tr>";
    }
    echo "</table>";
    }
    } else
    {
        echo "No record found!";
    }
    }
?>

