
var val ='tall than';
db.relationships_cannon.find({'relationships.predicate':val},{relationships:{$elemMatch:{predicate:val}}});
