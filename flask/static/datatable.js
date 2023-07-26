$(document).ready(function() {
    // Setup - add a text input to each footer cell
    $('#df_output tfoot th').each( function () {
        var title = $('#df_output thead th').eq( $(this).index() ).text();
        $(this).html( '<input type="text" placeholder="Search '+title+'" />' );
    } );

    // DataTable
    var table = $('#df_output').DataTable();

    // Apply the filter
    $("#df_output tfoot input").on( 'keyup change', function () {
        table
            .column( $(this).parent().index()+':visible' )
            .search( this.value )
            .draw();
} );
} );
