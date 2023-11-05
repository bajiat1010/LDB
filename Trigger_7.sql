create database database7;
use database7;
create TABLE CustomerAndSuppliers
(
cusl_id CHAR(6) PRIMARy KEY CHECK (cusl_id LIKE('[CS][0-9][0-9][0-9][0-9][0-9]')),
cusl_fname CHAR(15) NOT NULL,
cusl_lname VARCHAR (15),
cusl_address TEXT,
cusl_telno CHAR(12) CHECK(cusl_telno LIKE('[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]')),
cusl_city CHAR(12) DEFAULT 'Rajshahi',
sales_amnt MONEY CHECK(sales_amnt>=0),
proc_amnt MONEY CHECK(proc_amnt>=0)
)
select * from customerAndsuppliers
INSERT customerAndsuppliers
(cusl_id,cusl_fname,cusl_lname,cusl_address,cusl_telno,cusl_city,sales_amnt,proc_amnt) VALUES
('S00003','tofail','ahmed','kushtia,kushtia ','017-22633140','Kushtia',7,19)
----insert default value
---or /*

INSERT CustomerAndSuppliers
(cusl_id,cusl_fname,cusl_lname,cusl_address,cusl_telno,sales_amnt,proc_amnt) VALUES
('C00007','Iqbal','Hossain','221/B Dhanmondi','017-00000000',10,10)

create table Item
(
item_id char(6) primary key check (item_id like ('[P][0-9][0-9][0-9][0-9][0-9]')),
item_name char(12),
item_catagory char(10),
item_price float check(item_price>=0),
item_qoh int check(item_qoh>=0),
item_last_sold datetime  default getdate()
)
insert Item(item_id ,item_name,item_catagory,item_price ,item_qoh ,item_last_sold)
values('P00009','tofail','phone',57.8,23,'3-2-15')
insert Item(item_id ,item_name,item_catagory,item_price ,item_qoh )
values('P00002','tofail ah','laptop',56.8,23)
select * from Item

create table Transactions
(
tran_id char(10) primary key check (tran_id like '[T][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'),
item_id char(6) foreign key references Item(item_id), 
cusl_id char(6) foreign key references customerAndsuppliers(cusl_id) ,
tran_type char(1),
tran_quality int check (tran_quality>=0),
tran_date datetime default getdate()

)
select * from Transactions;
insert transactions(tran_id ,item_id,cusl_id,tran_type,tran_quality) values('T000000001','P00002','C00007','s',10)



select * from item

insert Item(item_id ,item_name,item_catagory,item_price ,item_qoh )
values('P00011','tofail b','Tab',14,30)

create trigger trg_test on item for insert 
as 
 begin
 print'one item insert into item table'
end

drop trigger trg_test
select * from transactions

CREATE TRIGGER trg_update_item ON Transactions FOR INSERT
AS
BEGIN
DECLARE @item_id char(6), @tranamount int, @tran_type char(1),@cusld_id char(6),@price money
SELECT @item_id=item_id, @tranamount=tran_quality, @tran_type=tran_type,@cusld_id =cusl_id FROM INSERTED
Select @price=item_price from item where item_id=@item_id
 IF (@tran_type ='S')
     begin
       UPDATE Item SET item_qoh=item_qoh- @tranamount WHERE item_id=@item_id 
       update  CustomerAndSuppliers set sales_amnt=sales_amnt+@price*@tranamount where cusl_id=@cusld_id
     end
 ELSE
   begin
    UPDATE Item SET item_qoh=item_qoh+ @tranamount WHERE item_id=@item_id
    update  CustomerAndSuppliers set proc_amnt=proc_amnt+@tranamount*@price where cusl_id=@cusld_id
    end
END

drop  TRIGGER trg_update_item

insert transactions(tran_id ,item_id,cusl_id,tran_type,tran_quality)
 values('T000000006','P00002','S00003','S',1)


select * from transactions
select * from CustomerAndSuppliers
select * from item


--show the trigger name  command
--select * from sys.triggers
--SELECT *  FROM sys.procedures



CREATE TRIGGER trg_update_item ON Transactions FOR INSERT
AS
BEGIN
DECLARE @item_id char(6), @tranamount int, @tran_type char(1)
SELECT @item_id=item_id, @tranamount=tran_quality, @tran_type=tran_type FROM INSERTED
IF (@tran_type ='S')
UPDATE Item SET item_qoh=item_qoh- @tranamount WHERE item_id=@item_id
ELSE
UPDATE Item SET item_price=item_price+ @tranamount WHERE item_id=@item_id
END

drop trigger trg_update_item

insert transactions(tran_id ,item_id,cusl_id,tran_type,tran_quality)
 values('T000000002','P00002','S00003','S',1)

select * from transactions
select * from item