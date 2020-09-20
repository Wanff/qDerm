express = require('express')
var app = express()
const port = process.env.PORT || 3000

const {spawn} = require('child_process');
var bodyParser = require('body-parser')
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({extended: true}))

app.use(express.static(__dirname + '/public'));
app.set("view engine", "ejs");

app.get("/", (req, res)=>{
    res.render('index')
})

app.get("/qderm", (req,res)=>{
    res.render('qml')
})

app.post("/qderm", calculateRisk)

function calculateRisk(req, res){
    var image = req.body.image
    var process = spawn('python',["./qml/qml.py", 
    image
]); 

    process.stdout.on('data', function(data) { 
        res.render('qml', {data: data.toString(), profile: req.body.image }); 
    })
}

app.listen(port, function(){
    console.log('Server listening on port ' + port)
})