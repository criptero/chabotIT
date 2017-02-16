if (annyang) {
	var commands = {
		'siguiente': function () {
			siguiente();
		},
	   
		'anterior': function () {
			anterior();
		},
	};
	annyang.setLanguage('es-ES');
	annyang.addCommands(commands);
	annyang.debug();
	annyang.start({ continuous: false });
}
