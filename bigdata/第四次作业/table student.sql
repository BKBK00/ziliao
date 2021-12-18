 create table student(
 student_id int,
 subject varchar(20),
 score int
 );

insert into student (student_id,subject,score) 
values 
(100,'数学',100),
(100,'语文',80),
(100,'理科',80),
(200,'数学',80),
(200,'语文',95),
(300,'数学',40),
(300,'语文',90),
(300,'社会',55),
(400,'数学',80);
SHOW CREATE TABLE student
SELECT * from student

SELECT student_id FROM student WHERE subject='语文' AND score>80