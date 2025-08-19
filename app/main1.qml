import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    id: win
    visible: true
    width: 1200
    height: 720
    title: "Zoo Classifier (Qt Quick)"

    Loader {
        id: mainLoader
        anchors.fill: parent
        source: "SplashPage.qml"
        onLoaded: {
            if (mainLoader.item && mainLoader.item.finished) {
                mainLoader.item.finished.connect(splashFinished)  // ✅ bağla
            }
        }
    }

    function splashFinished() {
        mainLoader.source = "MainPage.qml"
    }
}