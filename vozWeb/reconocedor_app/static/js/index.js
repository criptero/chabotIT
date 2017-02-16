var SW9  = undefined;
var table = undefined

$(function(){


});

var sendTranscription = function(transcription){
	$("#loader").html("<img src='/static/images/preload.gif'></img>")
	var url = '/vozWeb/getResultsAjax/?transcription='+transcription
	console.log(url)
	if ( !url )
        
		return null 
	$.ajax({
		
		url: url,
		type: "get",
		success : function(data)
		{

			console.log(data)
			PintaEnTabla(data);
			$("#loader").empty();
		},
		error: function(xhr,ajaxOptions,thrownError){
		}
	});
}

function PintaEnTabla(data_table) {
	if(table !== undefined){
		table.destroy()
	}
   	table = $('#data_table').DataTable( {
			data: data_table.data,
			columns: [
				{ title: "Título" , 	"orderable": true },
				{ title: "Descripción" , 	"orderable": false },
				{ title: "Requerimientos" , 	"orderable": false }
			],
			"order": [[ 0, "desc" ]],
			"pageLength": 5,
			"lengthMenu": [5, 10, 20, 50],
			"oLanguage":{
				"sProcessing":     "Procesando...",
		        "sLengthMenu":     "Mostrar _MENU_ registros",
		        "sZeroRecords":    "No se encontraron resultados",
		        "sEmptyTable":     "Ningún dato disponible en esta tabla",
		        "sInfo":           "Mostrando registros del _START_ al _END_ de un total de _TOTAL_ registros",
		        "sInfoEmpty":      "Mostrando registros del 0 al 0 de un total de 0 registros",
		        "sInfoFiltered":   "(filtrado de un total de _MAX_ registros)",
		        "sInfoPostFix":    "",
		        "sSearch":         "Buscar:",
		        "sUrl":            "",
		        "sInfoThousands":  ",",
		        "sLoadingRecords": "Cargando...",
		        "oPaginate": {
		               "sFirst":    "Primero",
		               "sLast":     "Último",
		               "sNext":     "Siguiente",
		               "sPrevious": "Anterior"
		        },
		        "oAria": {
		               "sSortAscending":  ": Activar para ordenar la columna de manera ascendente",
		               "sSortDescending": ": Activar para ordenar la columna de manera descendente"
		        }

			}

		} );
   	contarResultados(data_table.data.length)
}





