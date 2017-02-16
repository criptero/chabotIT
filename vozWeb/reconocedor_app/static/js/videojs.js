function mover_video(video){
    video.play();
}

function esperar_video(responsiveVoice){
	while(responsiveVoice.isPlaying()){
	}
}

function parar_video(){
	var video = document.getElementById("Video1");
    video.pause();
    button.textContent = ">";
   
	
}
