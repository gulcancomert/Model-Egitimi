import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Effects

Rectangle {
    id: splash
    anchors.fill: parent

    signal finished   // ✅ yeni sinyal

    property real anim: 0.0

    gradient: Gradient {
        GradientStop { position: 0.0; color: Qt.rgba(0.2 + anim, 0.3, 0.6, 1) }
        GradientStop { position: 1.0; color: Qt.rgba(1.0 - anim, 0.6, 0.9, 1) }
    }

    NumberAnimation on anim {
        from: 0; to: 0.8
        duration: 4000
        loops: Animation.Infinite
        easing.type: Easing.InOutQuad
    }

    Label {
        anchors.centerIn: parent
        text: "Hoşgeldiniz!"
        font.pixelSize: 42
        font.bold: true
        color: "white"

        SequentialAnimation on scale {
            loops: Animation.Infinite
            NumberAnimation { from: 1.0; to: 1.2; duration: 1200; easing.type: Easing.InOutQuad }
            NumberAnimation { from: 1.2; to: 1.0; duration: 1200; easing.type: Easing.InOutQuad }
        }
    }

    Timer {
        interval: 3000
        running: true
        repeat: false
        onTriggered: {
            splash.finished()   // ✅ sinyali gönder
        }
    }
}