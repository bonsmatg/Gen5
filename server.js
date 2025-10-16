const express = require('express');
const app = express();
app.use(express.static('dashboard'));
app.listen(3000, () => console.log('Jarvis Gen5 Dashboard running at http://localhost:3000'));