import MySQLdb


def make_sql_sentence(sql_sentence):
	results = []
	connection = MySQLdb.connect("localhost", "root", "everis", "ofertasDB")
	cursor = connection.cursor()
	try:
		cursor.execute("SET CHARACTER SET 'utf8'");
		cursor.execute('set names utf8')
		cursor.execute(sql_sentence)
		connection.commit()
		if cursor != None:
			for row in cursor:
				resultRow = []
				resultRow.append("<a  target='_blank' href='"+row[3].decode("utf-8")+"'>"+row[0].decode("utf-8")+"</a>")
				resultRow.append(row[1].decode("utf-8")[0:200]+ "... <img src='/static/images/sound.png' onclick='LeeDetalleOferta(\""+row[1].decode("utf-8")+"\")' style='width:20px;'>")
				resultRow.append(row[2].decode("utf-8")[0:200]+"...")
				results.append(resultRow)
		print " RESULTS : " + str(results)
		return results
	except:
		print " RESULTS : " + str(results)
		return results
		#outfile = open(variables.PATH_FILE_LOG, 'a')
		#outfile.write("ERROR AL INSERTAR\n")
		#outfile.write(sql_sentence)
	connection.close()
