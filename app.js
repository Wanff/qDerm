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
    var profile = req.body.profile
    var process = spawn('python',["./qml.py", 
    profile.gender, 
    profile.age,
    profile.race,
    profile.obese,
    profile.avg_salt_table,
    profile.everyday_cig,
    profile.smoking_env_people,
    profile.smoking_env_smokers,
    profile.smoking_env_days,
    profile.vig_work_freq,
    profile.mod_work_freq,
    profile.bike_walk_freq,
    profile.vig_play_freq,
    profile.mod_play_freq
]); 

    process.stdout.on('data', function(data) { 
        res.render('qml', {data: data.toString(), profile: req.body.profile }); 
    })
}

app.listen(port, function(){
    console.log('Server listening on port ' + port)
})