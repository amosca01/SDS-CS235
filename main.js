

// Read in schedule 
d3.csv("assets/sched.csv").then((data) => {

    let columns = data.columns; 
    
    //set up table
    let table = d3.select('#schedule')
                    .append('table') 
                        .attr('class', 'w3-table w3-bordered w3-border');

    let cols = table.selectAll('col')
                    .data(columns)
                    .enter()
                    .append('col')
                        .attr('class', (col) => {
                            if (col.includes('class')) {
                                return 'hide'; 
                            } 
                        }); 

    let thead = table.append('thead')
                        .attr('class', 'w3-metro-dark-blue'); 
    let tbody = table.append('tbody'); 

    //set header
    thead.append('tr')
            .selectAll('th')
            .data(columns)
            .enter()
            .append('th')
                .text((col) => { return col; }); 

    //add rows 
    let rows = tbody.selectAll('tr')
                    .data(data)
                    .enter()
                    .append('tr')
                        .attr('class', (d) => {
                            return d['class-row']; 
                        })
                        .attr('id', (d, i) => { return i; }); 

    //add cells 
    let top_clist = '';
    let cells = rows.selectAll('td')
                    .data((cells) => {
                        return columns.map( (col) => {
                            return {column: col, value: cells[col]}
                        })
                    })
                    .enter()
                    .append('td')
                        .attr('class', (d) => { 
                            if (d.column.includes('class-topic')) {
                                    top_clist = d.value; 
                            }  
                            if (d.column.includes('Topic')) {
                                return top_clist; 
                            } 
                        })
                        .text((d) => {  
                            return d.value; 
                        })
}); 













