import sqlite3

conn = sqlite3.connect('diauser.db')
print("Opened database successfully")

#conn.execute('CREATE TABLE diabetes (name VARCHAR(20)not null, phono VARCHAR(10) not null, email VARCHAR(50) not null,username VARCHAR(20) not null,password VARCHAR(20) not null)')
#print("Table created successfully")

cur = conn.cursor()
cur.execute('select * from diabetes');
account = cur.fetchall()
print(account)
conn.close()
