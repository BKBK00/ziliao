 create table DistrictProducts(
 district VARCHAR(20),
 name VARCHAR(20),
 price int
 );
insert into DistrictProducts (district,name,price)
values 
('东北','橘子',100),
('东北','苹果',50),
('东北','葡萄',50),
('东北','柠檬',30),
('关东','柠檬',100),
('关东','菠萝',100),
('关东','苹果',100),
('关东','葡萄',70),
('关西','柠檬',70),
('关西','西瓜',30),
('关西','苹果',20);
SELECT * from DistrictProducts

SELECT  D1.district,
        D1.name,
        D1.price,
        (SELECT count(D2.price)
           FROM DistrictProducts D2
           WHERE D2.price>D1.price and D2.district=D1.district)+1 AS rank_1
    from DistrictProducts D1
    order by D1.district , rank_1

